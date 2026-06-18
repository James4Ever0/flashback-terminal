package session

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"flashback-shell-pty/pkg/config"
	"flashback-shell-pty/pkg/log"
	"flashback-shell-pty/pkg/ptywrap"
	"flashback-shell-pty/pkg/vtcapture"
)

// Server owns a PTY/VT session and exposes it over a Unix socket.
type Server struct {
	cfg      *config.Config
	logger   *log.Logger
	sessionID string
	socketPath string
	cmd       *exec.Cmd
	ptmx      *os.File
	vtTerm    *vtcapture.Terminal

	listener net.Listener
	mu       sync.Mutex
	attached net.Conn
	attachedCh chan struct{} // closed when attached changes
	ready      chan struct{} // closed when the server is initialized

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	exitErr error
	exitCode int

	// background capture state
	lastCaptureHash string

	// counters
	vtDroppedBytes   atomic.Int64
	vtForwardedBytes atomic.Int64
}

// NewServer creates a new session server. It does not start the server.
func NewServer(cfg *config.Config, logger *log.Logger, sessionID, socketPath, cwd, shellBin string, shellArgs []string) (*Server, error) {
	cmd := exec.Command(shellBin, shellArgs...)
	cmd.Dir = cwd
	env := os.Environ()
	if os.Getenv("TERM") == "" {
		env = append(env, "TERM=xterm-256color")
	}
	env = setEnv(env, "FLASHBACK_SHELL", "1")
	cmd.Env = env

	ctx, cancel := context.WithCancel(context.Background())
	return &Server{
		cfg:        cfg,
		logger:     logger,
		sessionID:  sessionID,
		socketPath: socketPath,
		cmd:        cmd,
		ctx:        ctx,
		cancel:     cancel,
		attachedCh: make(chan struct{}),
		ready:      make(chan struct{}),
	}, nil
}

// Run starts the PTY, VT emulator, socket listener, and all goroutines, then
// blocks until the shell exits.
func (s *Server) Run() error {
	if err := os.MkdirAll(filepath.Dir(s.socketPath), 0755); err != nil {
		return fmt.Errorf("create socket dir: %w", err)
	}
	_ = os.Remove(s.socketPath)

	ptmx, err := ptywrap.Start(s.cmd)
	if err != nil {
		return fmt.Errorf("start pty: %w", err)
	}
	s.ptmx = ptmx

	s.vtTerm = vtcapture.NewTerminal(80, 24)
	s.vtTerm.SetScrollbackSize(s.cfg.ScrollbackLines)
	close(s.ready)

	l, err := net.Listen("unix", s.socketPath)
	if err != nil {
		s.ptmx.Close()
		return fmt.Errorf("listen %s: %w", s.socketPath, err)
	}
	s.listener = l

	s.logger.Infof("session server started: %s on %s", s.sessionID, s.socketPath)

	// PTY output -> VT emulator + attached client.
	s.wg.Add(1)
	go s.ptmxReader()

	// VT emulator input pipe -> PTY.
	s.wg.Add(1)
	go s.vtInputPipeForwarder()

	// Background capture.
	if !s.cfg.DisableCapture && s.cfg.CaptureInterval > 0 {
		s.wg.Add(1)
		go s.backgroundCapture()
	}

	// Socket listener.
	s.wg.Add(1)
	go s.socketListener()

	// Wait for shell exit.
	waitDone := make(chan struct{})
	go func() {
		s.exitErr = s.cmd.Wait()
		if s.exitErr != nil {
			if ee, ok := s.exitErr.(*exec.ExitError); ok {
				s.exitCode = ee.ExitCode()
			} else {
				s.exitCode = 1
			}
		}
		close(waitDone)
	}()

	// Signal handling for clean shutdown.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	select {
	case sig := <-sigCh:
		s.logger.Infof("session server received signal %s", sig)
		s.killShell()
		<-waitDone
	case <-waitDone:
	}

	// Brief grace period so a client that started just after the shell can
	// still attach and receive the final screen snapshot.
	s.mu.Lock()
	hasAttached := s.attached != nil
	s.mu.Unlock()
	if !hasAttached {
		select {
		case <-time.After(500 * time.Millisecond):
		case sig := <-sigCh:
			s.logger.Infof("session server received signal %s during grace", sig)
		}
	}

	s.shutdown()
	return s.exitErr
}

func (s *Server) ptmxReader() {
	defer s.wg.Done()
	// Buffered channel decouples live output from VT parser.
	vtInput := make(chan []byte, 256)

	// VT emulator processor.
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		var totalProcessed int64
		for data := range vtInput {
			done := make(chan struct{})
			go func() {
				_, _ = s.vtTerm.Write(data)
				close(done)
			}()
			select {
			case <-done:
				totalProcessed += int64(len(data))
			case <-time.After(500 * time.Millisecond):
				s.logger.Warnf("VT emulator write timed out, disabling VT captures")
				for range vtInput {
				}
				return
			}
		}
	}()

	buf := make([]byte, 4096)
	for {
		n, err := s.ptmx.Read(buf)
		if n > 0 {
			data := append([]byte(nil), buf[:n]...)
			s.writeToAttached(data)
			select {
			case vtInput <- data:
			default:
				s.vtDroppedBytes.Add(int64(n))
			}
		}
		if err != nil {
			close(vtInput)
			return
		}
	}
}

func (s *Server) vtInputPipeForwarder() {
	defer s.wg.Done()
	buf := make([]byte, 4096)
	for {
		n, err := s.vtTerm.Read(buf)
		if n > 0 {
			s.vtForwardedBytes.Add(int64(n))
			_, _ = s.ptmx.Write(buf[:n])
		}
		if err != nil {
			return
		}
	}
}

func (s *Server) socketListener() {
	defer s.wg.Done()
	for {
		conn, err := s.listener.Accept()
		if err != nil {
			select {
			case <-s.ctx.Done():
				return
			default:
				s.logger.Warnf("accept error: %v", err)
				continue
			}
		}
		s.wg.Add(1)
		go func() {
			defer s.wg.Done()
			s.handleConn(conn)
		}()
	}
}

func (s *Server) handleConn(conn net.Conn) {
	defer conn.Close()
	reader := bufio.NewReader(conn)
	req, err := ReadRequest(reader)
	if err != nil {
		s.logger.Debugf("failed to read request: %v", err)
		return
	}

	switch req.Method {
	case MethodAttach:
		s.handleAttach(conn, reader, req)
	case MethodCapture:
		s.handleCapture(conn, req)
	case MethodStatus:
		s.handleStatus(conn)
	case MethodKill:
		s.handleKill(conn)
	case MethodResize:
		s.handleResize(conn, req)
	default:
		_ = WriteResponse(conn, ErrorResponse(req.Method, fmt.Errorf("unknown method %q", req.Method)))
	}
}

func (s *Server) handleAttach(conn net.Conn, reader *bufio.Reader, req *Request) {
	s.mu.Lock()
	if s.attached != nil {
		s.mu.Unlock()
		_ = WriteResponse(conn, ErrorResponse(MethodAttach, fmt.Errorf("session already attached")))
		return
	}
	s.attached = conn
	s.mu.Unlock()
	s.logger.Infof("client attached to session %s", s.sessionID)

	if req.Cols > 0 && req.Rows > 0 {
		s.resize(req.Cols, req.Rows)
	}

	// Print the current screen snapshot so the user sees output that was
	// produced before the attach completed.
	screen := strings.TrimRight(string(s.vtTerm.CaptureANSI()), " \t\r\n")
	if screen != "" {
		screen += "\n"
	}

	cols, rows := s.vtTerm.Size()
	resp, err := SuccessResponse(MethodAttach, &StatusPayload{
		SessionID: s.sessionID,
		ShellPID:  s.cmd.Process.Pid,
		Cols:      cols,
		Rows:      rows,
		Attached:  true,
		Screen:    screen,
	})
	if err != nil {
		_ = WriteResponse(conn, ErrorResponse(MethodAttach, err))
		s.detach(conn)
		return
	}
	if err := WriteResponse(conn, resp); err != nil {
		s.logger.Warnf("failed to send attach response: %v", err)
		s.detach(conn)
		return
	}

	// Forward client -> PTY until disconnect or server shutdown.
	buf := make([]byte, 4096)
	for {
		n, err := conn.Read(buf)
		if n > 0 {
			_, _ = s.ptmx.Write(buf[:n])
		}
		if err != nil {
			break
		}
	}

	s.detach(conn)
	s.logger.Infof("client detached from session %s", s.sessionID)
	s.killShell()
}

func (s *Server) handleCapture(conn net.Conn, req *Request) {
	ansi, text, cols, rows, row, col := s.CaptureScreen(req.Scrollback)

	payload := CapturePayload{
		ANSI:      ansi,
		Text:      text,
		Cols:      cols,
		Rows:      rows,
		CursorRow: row,
		CursorCol: col,
	}
	resp, err := SuccessResponse(MethodCapture, payload)
	if err != nil {
		_ = WriteResponse(conn, ErrorResponse(MethodCapture, err))
		return
	}
	_ = WriteResponse(conn, resp)
}

// Ready returns a channel that is closed once the server has initialized
// the PTY and VT emulator and is ready to accept commands.
func (s *Server) Ready() <-chan struct{} {
	return s.ready
}

// CaptureScreen returns the current screen content. It is safe to call from
// outside the server's goroutines.
func (s *Server) CaptureScreen(scrollback bool) (ansi, text string, cols, rows, cursorRow, cursorCol int) {
	var a, t []byte
	if scrollback {
		a = s.vtTerm.CaptureScrollbackANSI(s.cfg.ScrollbackLines)
		t = s.vtTerm.CaptureScrollbackText(s.cfg.ScrollbackLines)
	} else {
		a = s.vtTerm.CaptureANSI()
		t = s.vtTerm.CaptureText()
	}
	cols, rows = s.vtTerm.Size()
	cursorRow, cursorCol = s.vtTerm.Cursor()
	return string(a), string(t), cols, rows, cursorRow, cursorCol
}

func (s *Server) handleStatus(conn net.Conn) {
	cols, rows := s.vtTerm.Size()
	s.mu.Lock()
	attached := s.attached != nil
	s.mu.Unlock()

	payload := StatusPayload{
		SessionID: s.sessionID,
		ShellPID:  s.cmd.Process.Pid,
		Cols:      cols,
		Rows:      rows,
		Attached:  attached,
	}
	resp, err := SuccessResponse(MethodStatus, payload)
	if err != nil {
		_ = WriteResponse(conn, ErrorResponse(MethodStatus, err))
		return
	}
	_ = WriteResponse(conn, resp)
}

func (s *Server) handleKill(conn net.Conn) {
	resp, err := SuccessResponse(MethodKill, &StatusPayload{SessionID: s.sessionID})
	if err != nil {
		_ = WriteResponse(conn, ErrorResponse(MethodKill, err))
		return
	}
	_ = WriteResponse(conn, resp)
	s.killShell()
}

func (s *Server) handleResize(conn net.Conn, req *Request) {
	s.resize(req.Cols, req.Rows)
	cols, rows := s.vtTerm.Size()
	resp, err := SuccessResponse(MethodResize, &StatusPayload{
		SessionID: s.sessionID,
		ShellPID:  s.cmd.Process.Pid,
		Cols:      cols,
		Rows:      rows,
	})
	if err != nil {
		_ = WriteResponse(conn, ErrorResponse(MethodResize, err))
		return
	}
	_ = WriteResponse(conn, resp)
}

func (s *Server) resize(cols, rows int) {
	if cols <= 0 || rows <= 0 {
		return
	}
	_ = ptywrap.Resize(s.ptmx, uint16(rows), uint16(cols))
	s.vtTerm.Resize(cols, rows)
	s.logger.Debugf("resized session %s to %dx%d", s.sessionID, cols, rows)
}

func (s *Server) writeToAttached(data []byte) {
	s.mu.Lock()
	conn := s.attached
	s.mu.Unlock()
	if conn == nil {
		return
	}
	if _, err := conn.Write(data); err != nil {
		s.logger.Debugf("write to attached client failed: %v", err)
		s.detach(conn)
	}
}

func (s *Server) detach(conn net.Conn) {
	s.mu.Lock()
	if s.attached == conn {
		s.attached = nil
	}
	s.mu.Unlock()
}

func (s *Server) killShell() {
	if s.cmd.Process != nil {
		_ = s.cmd.Process.Kill()
	}
}

func (s *Server) shutdown() {
	s.cancel()
	if s.listener != nil {
		s.listener.Close()
	}
	if s.ptmx != nil {
		s.ptmx.Close()
	}
	if s.vtTerm != nil {
		_ = s.vtTerm.Close()
	}
	s.mu.Lock()
	if s.attached != nil {
		_ = s.attached.Close()
		s.attached = nil
	}
	s.mu.Unlock()
	_ = os.Remove(s.socketPath)
	s.wg.Wait()
	s.logger.Infof("session server %s shut down (exit code %d)", s.sessionID, s.exitCode)
}

func (s *Server) backgroundCapture() {
	defer s.wg.Done()

	home, _ := os.UserHomeDir()
	stateDir := filepath.Join(home, ".flashback-shell-pty", "state")
	_ = os.MkdirAll(stateDir, 0755)

	interval := time.Duration(s.cfg.CaptureInterval) * time.Second
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	select {
	case <-s.ctx.Done():
		return
	case <-time.After(time.Duration(s.cfg.FirstCaptureDelay) * time.Second):
	}
	s.doCapture(stateDir)

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.doCapture(stateDir)
		}
	}
}

func (s *Server) doCapture(stateDir string) {
	// This is a lightweight inline capture for background uploads.
	// Full upload buffering is handled by the explicit capture command.
	if s.cfg.ServerURL == "" {
		return
	}

	ansi := s.vtTerm.CaptureANSI()
	text := s.vtTerm.CaptureText()
	if s.cfg.TextOnly {
		ansi = nil
	}

	hashInput := string(ansi)
	if s.cfg.TextOnly {
		hashInput = string(text)
	}
	if hashInput == "" {
		return
	}

	h := hashString(hashInput)
	if h == s.lastCaptureHash {
		return
	}
	s.lastCaptureHash = h

	cols, rows := s.vtTerm.Size()
	row, col := s.vtTerm.Cursor()

	s.logger.Debugf("background capture: %dx%d hash=%s", cols, rows, h)
	_ = os.WriteFile(filepath.Join(stateDir, s.sessionID+".hash"), []byte(h), 0644)

	// TODO: upload via pkg/server client once available.
	_ = text
	_ = row
	_ = col
}

func hashString(s string) string {
	// Simple stable hash; replace with MD5 in capture engine.
	var h uint32
	for _, c := range s {
		h = h*31 + uint32(c)
	}
	return fmt.Sprintf("%08x", h)
}

// setEnv returns a copy of env with key=value set, replacing any existing
// entry with the same key.
func setEnv(env []string, key, value string) []string {
	prefix := key + "="
	for i, e := range env {
		if strings.HasPrefix(e, prefix) {
			env[i] = prefix + value
			return env
		}
	}
	return append(env, prefix+value)
}
