package cmd

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"flashback-shell-pty/pkg/capture"
	"flashback-shell-pty/pkg/config"
	"flashback-shell-pty/pkg/log"
	"flashback-shell-pty/pkg/ptywrap"
	"flashback-shell-pty/pkg/rawmode"
	"flashback-shell-pty/pkg/server"
	"flashback-shell-pty/pkg/vtcapture"
)

// NewCmd handles the "new" subcommand.
func NewCmd(cfg *config.Config, logger *log.Logger, args []string) int {
	var (
		captureFile     string
		captureTextFile string
		vtLogFile       string
		vtLogInterval   = 2 * time.Second
	)

	// Parse known subcommand flags from the front of args, leaving the rest as
	// arguments for the shell. Stop at the first non-flag argument or an
	// explicit "--" separator.
	i := 0
parseLoop:
	for i < len(args) {
		arg := args[i]
		if arg == "--" {
			i++
			break
		}
		if !strings.HasPrefix(arg, "-") {
			break
		}
		// Long flags only.
		if !strings.HasPrefix(arg, "--") {
			// First short-looking argument is treated as a shell argument.
			break
		}
		switch {
		case arg == "--capture":
			if i+1 >= len(args) {
				logger.Errorf("missing value for --capture")
				return 1
			}
			captureFile = args[i+1]
			i += 2
		case strings.HasPrefix(arg, "--capture="):
			captureFile = strings.TrimPrefix(arg, "--capture=")
			i++
		case arg == "--capture-text":
			if i+1 >= len(args) {
				logger.Errorf("missing value for --capture-text")
				return 1
			}
			captureTextFile = args[i+1]
			i += 2
		case strings.HasPrefix(arg, "--capture-text="):
			captureTextFile = strings.TrimPrefix(arg, "--capture-text=")
			i++
		case arg == "--vt-log":
			if i+1 >= len(args) {
				logger.Errorf("missing value for --vt-log")
				return 1
			}
			vtLogFile = args[i+1]
			i += 2
		case strings.HasPrefix(arg, "--vt-log="):
			vtLogFile = strings.TrimPrefix(arg, "--vt-log=")
			i++
		case arg == "--vt-log-interval":
			if i+1 >= len(args) {
				logger.Errorf("missing value for --vt-log-interval")
				return 1
			}
			d, err := time.ParseDuration(args[i+1])
			if err != nil {
				logger.Errorf("invalid --vt-log-interval: %v", err)
				return 1
			}
			vtLogInterval = d
			i += 2
		case strings.HasPrefix(arg, "--vt-log-interval="):
			d, err := time.ParseDuration(strings.TrimPrefix(arg, "--vt-log-interval="))
			if err != nil {
				logger.Errorf("invalid --vt-log-interval: %v", err)
				return 1
			}
			vtLogInterval = d
			i++
		default:
			// Unknown long flag: treat it and the rest as shell args.
			break parseLoop
		}
	}
	shellArgs := args[i:]

	if len(shellArgs) > 0 && strings.TrimLeft(shellArgs[0], "-") == "no-capture" {
		logger.Errorf("--no-capture is a global flag and must appear before the subcommand (e.g. flashback-shell-pty --no-capture new)")
		return 1
	}

	if os.Getenv("FLASHBACK_SHELL") != "" && !cfg.AllowNested {
		logger.Errorf("refusing to start a nested flashback-shell session (FLASHBACK_SHELL is set); use --allow-nested or set allow_nested: true to override")
		return 1
	}

	logger.Infof("effective config: server_url=%s shell=%s buffer_size=%d buffer_mode=%s buffer_dir=%s device_id=%s capture_interval=%d first_capture_delay=%d disable_capture=%t diff_only=%t diff_mode=%s text_only=%t scrollback_lines=%d allow_nested=%t",
		cfg.ServerURL, cfg.Shell, cfg.BufferSize, cfg.BufferMode, cfg.BufferDir, cfg.DeviceID, cfg.CaptureInterval, cfg.FirstCaptureDelay, cfg.DisableCapture, cfg.DiffOnly, cfg.DiffMode, cfg.TextOnly, cfg.ScrollbackLines, cfg.AllowNested)

	shellBin := cfg.Shell
	if shellBin == "" {
		shellBin = os.Getenv("SHELL")
		if shellBin == "" {
			shellBin = "/bin/bash"
		}
	}

	if len(shellArgs) == 0 {
		shellArgs = []string{"-l"}
	}

	cmd := exec.Command(shellBin, shellArgs...)
	cmd.Env = os.Environ()
	if os.Getenv("TERM") == "" {
		cmd.Env = append(cmd.Env, "TERM=xterm-256color")
	}
	cmd.Env = setEnv(cmd.Env, "FLASHBACK_SHELL", "1")

	ptmx, err := ptywrap.Start(cmd)
	if err != nil {
		logger.Errorf("failed to start pty: %v", err)
		return 1
	}
	defer ptmx.Close()

	vtTerm := vtcapture.NewTerminal(80, 24)
	vtTerm.SetScrollbackSize(cfg.ScrollbackLines)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup

	// Set initial size from the local terminal.
	if rawmode.IsTerminal(int(os.Stdin.Fd())) {
		cols, rows, err := rawmode.Size(int(os.Stdin.Fd()))
		if err == nil && cols > 0 && rows > 0 {
			_ = ptywrap.Resize(ptmx, uint16(rows), uint16(cols))
			vtTerm.Resize(cols, rows)
		}
	}

	// Continuous VT capture log.
	var vtLogger *logBuf
	if vtLogFile != "" {
		f, err := os.Create(vtLogFile)
		if err != nil {
			logger.Errorf("vt log file: %v", err)
			return 1
		}
		defer f.Close()
		vtLogger = newLogBuf(f)
		wg.Add(1)
		go func() {
			defer wg.Done()
			ticker := time.NewTicker(vtLogInterval)
			defer ticker.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
					capture := vtTerm.CaptureANSI()
					cols, rows := vtTerm.Size()
					vtLogger.Writef("--- %s %dx%d ---\n", time.Now().Format(time.RFC3339Nano), cols, rows)
					vtLogger.Write(capture)
					vtLogger.WriteByte('\n')
					vtLogger.Flush()
				}
			}
		}()
	}

	// Signal handling for clean shutdown. SIGHUP is included so closing the
	// terminal window terminates the wrapper cleanly.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM, syscall.SIGHUP)
	defer signal.Stop(sigCh)

	// Raw mode.
	var restore func() error
	if rawmode.IsTerminal(int(os.Stdin.Fd())) {
		var err error
		restore, _, err = rawmode.MakeRaw(int(os.Stdin.Fd()))
		if err != nil {
			logger.Warnf("make raw: %v", err)
		}
	}
	if restore != nil {
		defer restore()
	}

	// Resize monitor.
	resizeCh := make(chan os.Signal, 1)
	signal.Notify(resizeCh, syscall.SIGWINCH)
	defer signal.Stop(resizeCh)
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-ctx.Done():
				return
			case <-resizeCh:
				if rawmode.IsTerminal(int(os.Stdin.Fd())) {
					cols, rows, err := rawmode.Size(int(os.Stdin.Fd()))
					if err == nil && cols > 0 && rows > 0 {
						_ = ptywrap.Resize(ptmx, uint16(rows), uint16(cols))
						vtTerm.Resize(cols, rows)
						logger.Debugf("resized to %dx%d", cols, rows)
					}
				}
			}
		}
	}()

	// Forward stdin -> PTY. This goroutine is intentionally not part of the
	// wait group: os.Stdin.Read can stay blocked after the child exits, and
	// waiting for it would prevent clean shutdown.
	go func() {
		buf := make([]byte, 4096)
		for {
			n, err := os.Stdin.Read(buf)
			if n > 0 {
				_, _ = ptmx.Write(buf[:n])
			}
			if err != nil {
				return
			}
		}
	}()

	// Counters for VT emulator health.
	var vtDroppedBytes atomic.Int64
	var vtForwardedBytes atomic.Int64

	// Forward PTY -> stdout and VT emulator.
	// VT processing is async with a timeout so that a blocked emulator
	// (e.g. its reply pipe full because nobody consumes it) does not freeze
	// live stdout output.
	wg.Add(1)
	go func() {
		defer wg.Done()
		vtInput := make(chan []byte, 256)

		wg.Add(1)
		go func() {
			defer wg.Done()
			for data := range vtInput {
				done := make(chan struct{})
				go func() {
					_, _ = vtTerm.Write(data)
					close(done)
				}()
				select {
				case <-done:
				case <-time.After(500 * time.Millisecond):
					logger.Warnf("VT emulator write timed out, disabling VT captures")
					for range vtInput {
					}
					return
				}
			}
		}()

		buf := make([]byte, 4096)
		for {
			n, err := ptmx.Read(buf)
			if n > 0 {
				_, _ = os.Stdout.Write(buf[:n])
				data := append([]byte(nil), buf[:n]...)
				select {
				case vtInput <- data:
				default:
					vtDroppedBytes.Add(int64(n))
				}
			}
			if err != nil {
				close(vtInput)
				return
			}
		}
	}()

	// VT emulator input pipe -> PTY.
	// The emulator writes reply sequences (e.g. OSC 10/11 color responses) to
	// an internal pipe; draining it and forwarding those bytes back to the PTY
	// prevents TUIs like vim from blocking on terminal queries.
	wg.Add(1)
	go func() {
		defer wg.Done()
		buf := make([]byte, 4096)
		for {
			n, err := vtTerm.Read(buf)
			if n > 0 {
				vtForwardedBytes.Add(int64(n))
				_, _ = ptmx.Write(buf[:n])
			}
			if err != nil {
				return
			}
		}
	}()

	// Background capture and upload.
	var captureCancel context.CancelFunc
	var captureWg sync.WaitGroup
	if !cfg.DisableCapture && cfg.CaptureInterval > 0 {
		var captureCtx context.Context
		captureCtx, captureCancel = context.WithCancel(ctx)
		captureWg.Add(1)
		go func() {
			defer captureWg.Done()
			backgroundCapture(captureCtx, cfg, vtTerm, logger)
		}()
	}

	// Wait for command exit in a goroutine so we can react to signals.
	waitDone := make(chan struct{})
	var exitErr error
	go func() {
		exitErr = cmd.Wait()
		close(waitDone)
	}()

	select {
	case sig := <-sigCh:
		logger.Infof("received signal %s, shutting down", sig)
		_ = cmd.Process.Kill()
		<-waitDone
	case <-waitDone:
	}

	status := 0
	if exitErr != nil {
		if ee, ok := exitErr.(*exec.ExitError); ok {
			status = ee.ExitCode()
		} else {
			status = 1
		}
	}

	if captureCancel != nil {
		captureCancel()
		captureWg.Wait()
	}

	// Final capture writes.
	if captureFile != "" {
		if err := os.WriteFile(captureFile, vtTerm.CaptureANSI(), 0644); err != nil {
			logger.Warnf("failed to write ansi capture: %v", err)
		}
	}
	if captureTextFile != "" {
		if err := os.WriteFile(captureTextFile, vtTerm.CaptureText(), 0644); err != nil {
			logger.Warnf("failed to write text capture: %v", err)
		}
	}

	if vtLogger != nil {
		vtLogger.Flush()
	}

	cancel()
	// Closing the VT emulator unblocks the input-pipe forwarder goroutine so
	// wg.Wait() can return promptly.
	_ = vtTerm.Close()
	wg.Wait()

	return status
}

func backgroundCapture(ctx context.Context, cfg *config.Config, vtTerm *vtcapture.Terminal, logger *log.Logger) {
	buffer, cleanup, err := newRetryBuffer(cfg)
	if err != nil {
		logger.Warnf("failed to create retry buffer: %v", err)
	}
	if cleanup != nil {
		defer cleanup()
	}

	client := server.NewClient(cfg, buffer, logger)

	// Flush any retries from a previous run. With the temp subfolder pattern,
	// previous data is already gone, so this is effectively a no-op.
	if buffer != nil {
		if err := client.FlushRetries(); err != nil {
			logger.Debugf("flush retries: %v", err)
		}
	}

	engine := capture.NewEngine("")
	engine.DiffOnly = cfg.DiffOnly
	engine.DiffMode = cfg.DiffMode
	engine.TextOnly = cfg.TextOnly

	do := func() {
		c := engine.CaptureScreen(vtTerm, cfg)
		if c == nil {
			return
		}
		logger.Infof("background capture: changed")
		if err := client.Upload([]capture.Capture{*c}); err != nil {
			logger.Debugf("background upload: %v", err)
		}
	}

	interval := time.Duration(cfg.CaptureInterval) * time.Second
	logger.Infof("background capture started (first_capture_delay=%s, interval=%s)", time.Duration(cfg.FirstCaptureDelay)*time.Second, interval)

	if cfg.FirstCaptureDelay > 0 {
		select {
		case <-ctx.Done():
			logger.Infof("background capture stopped")
			return
		case <-time.After(time.Duration(cfg.FirstCaptureDelay) * time.Second):
		}
	}

	do()

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			logger.Infof("background capture stopped")
			return
		case <-ticker.C:
			do()
		}
	}
}

func newRetryBuffer(cfg *config.Config) (*capture.Buffer, func(), error) {
	parent := cfg.BufferDir
	if parent == "" {
		parent = "/tmp"
	}
	dir, err := os.MkdirTemp(parent, ".flashback-shell-pty-buffer-*")
	if err != nil {
		return nil, nil, err
	}
	cleanup := func() {
		_ = os.RemoveAll(dir)
	}
	return capture.NewBuffer(dir, cfg.BufferSize), cleanup, nil
}

// logBuf wraps an os.File with a small buffer.
type logBuf struct {
	w   *os.File
	buf []byte
}

func newLogBuf(f *os.File) *logBuf {
	return &logBuf{w: f}
}

func (l *logBuf) Writef(format string, a ...interface{}) {
	l.buf = append(l.buf, []byte(fmt.Sprintf(format, a...))...)
}

func (l *logBuf) Write(p []byte) {
	l.buf = append(l.buf, p...)
}

func (l *logBuf) WriteByte(b byte) error {
	l.buf = append(l.buf, b)
	return nil
}

func (l *logBuf) Flush() {
	if len(l.buf) == 0 {
		return
	}
	_, _ = l.w.Write(l.buf)
	l.buf = l.buf[:0]
}

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
