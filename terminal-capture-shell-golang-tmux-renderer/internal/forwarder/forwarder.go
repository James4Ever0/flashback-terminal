// Package forwarder wraps a command in a PTY, mirrors the local terminal, and
// forwards typed JSON-RPC events to a Unix socket listener.
package forwarder

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"flashback-shell-tmux-renderer/internal/event"
	"flashback-shell-tmux-renderer/internal/log"
	"flashback-shell-tmux-renderer/internal/protocol"
	"flashback-shell-tmux-renderer/internal/ptywrap"
	"flashback-shell-tmux-renderer/internal/rawmode"
)

// Options configures a forwarder run.
type Options struct {
	Command    string
	Args       []string
	Dir        string
	SocketPath string
	Logger     *log.Logger
}

// Run starts the PTY forwarder with the given options and blocks until the
// wrapped command exits. It returns the command's exit status.
func Run(opts Options) int {
	logger := opts.Logger
	if logger == nil {
		logger = log.New(0, os.Stderr)
	}

	if opts.Command == "" {
		logger.Errorf("no command specified")
		return 1
	}

	cmd := exec.Command(opts.Command, opts.Args...)
	if opts.Dir != "" {
		cmd.Dir = opts.Dir
	}

	// Build the child's environment with the correct terminal size before starting.
	env := os.Environ()
	if rawmode.IsTerminal(int(os.Stdin.Fd())) {
		cols, rows, err := rawmode.Size(int(os.Stdin.Fd()))
		logger.Debugf("[env] rawmode.Size(fd=%d) = cols=%d rows=%d err=%v", int(os.Stdin.Fd()), cols, rows, err)
		if err == nil && cols > 0 && rows > 0 {
			env = setEnv(env, "COLUMNS", fmt.Sprintf("%d", cols))
			env = setEnv(env, "LINES", fmt.Sprintf("%d", rows))
			logger.Debugf("[env] set COLUMNS=%d LINES=%d", cols, rows)
		}
	}
	if os.Getenv("TERM") == "" {
		env = setEnv(env, "TERM", "xterm-256color")
		logger.Debugf("[env] set TERM=xterm-256color")
	}
	cmd.Env = env

	ptmx, err := ptywrap.Start(cmd)
	if err != nil {
		logger.Errorf("failed to start pty: %v", err)
		return 1
	}
	defer ptmx.Close()
	childPid := cmd.Process.Pid

	bus := event.NewBus()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	stdinDone := make(chan struct{})
	var wg sync.WaitGroup

	// Set initial size from the local terminal.
	if rawmode.IsTerminal(int(os.Stdin.Fd())) {
		cols, rows, err := rawmode.Size(int(os.Stdin.Fd()))
		logger.Debugf("[init resize] rawmode.Size(fd=%d) = cols=%d rows=%d err=%v", int(os.Stdin.Fd()), cols, rows, err)
		if err == nil && cols > 0 && rows > 0 {
			_ = ptywrap.Resize(ptmx, uint16(rows), uint16(cols))
			logger.Debugf("[init resize] ptywrap.Resize(rows=%d, cols=%d)", rows, cols)
			bus.SetScreen(uint16(cols), uint16(rows), nil)
			id, t := bus.NextID()
			logger.Debugf("[init resize] publish ResizeEvent{Cols=%d, Rows=%d}", cols, rows)
			bus.Publish(&event.ResizeEvent{Event: event.Event{ID: id, Time: t}, Cols: uint16(cols), Rows: uint16(rows)})
		}
	}

	// Listener setup.
	var listener net.Listener
	if opts.SocketPath != "" {
		if err := os.MkdirAll(filepath.Dir(opts.SocketPath), 0755); err != nil {
			logger.Errorf("create socket dir: %v", err)
			return 1
		}
		_ = os.Remove(opts.SocketPath)
		l, err := net.Listen("unix", opts.SocketPath)
		if err != nil {
			logger.Errorf("listen %s: %v", opts.SocketPath, err)
			return 1
		}
		listener = l
		wg.Add(1)
		go func() {
			defer wg.Done()
			serveListener(ctx, listener, bus, logger)
		}()
		logger.Infof("forwarder listening on %s", opts.SocketPath)
	}

	// Signal handling for clean shutdown.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	var restore func() error
	if rawmode.IsTerminal(int(os.Stdin.Fd())) {
		var err error
		restore, _, err = rawmode.MakeRaw(int(os.Stdin.Fd()))
		if err != nil {
			logger.Errorf("make raw: %v", err)
			return 1
		}
		defer func() {
			if restore != nil {
				_ = restore()
			}
		}()
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
					logger.Debugf("[sigwinch] rawmode.Size(fd=%d) = cols=%d rows=%d err=%v", int(os.Stdin.Fd()), cols, rows, err)
					if err == nil && cols > 0 && rows > 0 {
						_ = ptywrap.Resize(ptmx, uint16(rows), uint16(cols))
						id, t := bus.NextID()
						logger.Debugf("[sigwinch] publish ResizeEvent{Cols=%d, Rows=%d}", cols, rows)
						bus.Publish(&event.ResizeEvent{Event: event.Event{ID: id, Time: t}, Cols: uint16(cols), Rows: uint16(rows)})
						// Forward SIGWINCH so the child's process group re-reads its terminal size.
						if childPid > 0 {
							logger.Debugf("[sigwinch] forward SIGWINCH to child process group -%d", childPid)
							_ = syscall.Kill(-childPid, syscall.SIGWINCH)
						}
					}
				}
			}
		}
	}()

	// Forward stdin -> PTY. This goroutine is intentionally not part of wg:
	// after the child exits, os.Stdin.Read may stay blocked until the process
	// exits, so we must not wait for it.
	go func() {
		buf := make([]byte, 4096)
		for {
			select {
			case <-stdinDone:
				return
			default:
			}
			n, err := os.Stdin.Read(buf)
			if n > 0 {
				_, _ = ptmx.Write(buf[:n])
				id, t := bus.NextID()
				bus.Publish(&event.InputEvent{Event: event.Event{ID: id, Time: t}, Data: protocol.EncodeBytes(buf[:n])})
			}
			if err != nil {
				return
			}
		}
	}()

	// Forward PTY -> stdout and bus. Use a separate goroutine for stdout writes
	// so a slow/detached consumer cannot block ptmx reads or event publishing.
	stdoutCh := make(chan []byte, 64)
	wg.Add(1)
	go func() {
		defer wg.Done()
		for data := range stdoutCh {
			_, _ = os.Stdout.Write(data)
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		buf := make([]byte, 4096)
		for {
			n, err := ptmx.Read(buf)
			if n > 0 {
				data := append([]byte(nil), buf[:n]...)
				id, t := bus.NextID()
				bus.Publish(&event.OutputEvent{Event: event.Event{ID: id, Time: t}, Data: protocol.EncodeBytes(data)})
				select {
				case stdoutCh <- data:
				default:
					// Channel full: drop the local stdout write rather than block
					// the event bus. The socket subscribers still received the event.
				}
			}
			if err != nil {
				close(stdoutCh)
				return
			}
		}
	}()

	// Wait for command exit in a goroutine so we can react to signals.
	waitDone := make(chan struct{})
	var exitErr error
	go func() {
		exitErr = cmd.Wait()
		close(waitDone)
		close(stdinDone)
	}()

	cleanup := func() {
		if restore != nil {
			_ = restore()
			restore = nil
		}
		_ = syscall.Close(0)
	}

	select {
	case sig := <-sigCh:
		logger.Infof("received signal %s, forwarding to child process group", sig)
		if childPid > 0 {
			if s, ok := sig.(syscall.Signal); ok {
				_ = syscall.Kill(-childPid, s)
			}
		}
		select {
		case <-waitDone:
			cleanup()
		case <-time.After(5 * time.Second):
			logger.Warnf("child did not exit gracefully, sending SIGKILL")
			if childPid > 0 {
				_ = syscall.Kill(-childPid, syscall.SIGKILL)
			}
			<-waitDone
			cleanup()
		}
	case <-waitDone:
		cleanup()
	}

	status := 0
	if exitErr != nil {
		if ee, ok := exitErr.(*exec.ExitError); ok {
			status = ee.ExitCode()
		} else {
			status = 1
		}
	}

	id, t := bus.NextID()
	bus.Publish(&event.ExitEvent{Event: event.Event{ID: id, Time: t}, Status: status})
	id, t = bus.NextID()
	bus.Publish(&event.EofEvent{Event: event.Event{ID: id, Time: t}})

	// Give subscribers a moment to drain the final events.
	time.Sleep(100 * time.Millisecond)

	// Close the PTY master so the PTY->stdout goroutine sees EOF and exits.
	ptmx.Close()

	cancel()
	if listener != nil {
		listener.Close()
	}

	// Wait for the remaining goroutines, but do not block forever.
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		logger.Warnf("timeout waiting for forwarder goroutines to finish")
	}

	if opts.SocketPath != "" {
		_ = os.Remove(opts.SocketPath)
	}

	return status
}

func setEnv(env []string, key, value string) []string {
	prefix := key + "="
	for i, e := range env {
		if len(e) >= len(prefix) && e[:len(prefix)] == prefix {
			env[i] = prefix + value
			return env
		}
	}
	return append(env, prefix+value)
}

func serveListener(ctx context.Context, listener net.Listener, bus *event.Bus, logger *log.Logger) {
	for {
		conn, err := listener.Accept()
		if err != nil {
			select {
			case <-ctx.Done():
				return
			default:
				logger.Debugf("accept error: %v", err)
				continue
			}
		}
		go handleConn(ctx, conn, bus, logger)
	}
}

func handleConn(ctx context.Context, conn net.Conn, bus *event.Bus, logger *log.Logger) {
	defer conn.Close()
	subID, init, ch := bus.Subscribe()
	defer bus.Unsubscribe(subID)

	if err := writeEvent(conn, init); err != nil {
		logger.Debugf("write init event: %v", err)
		return
	}

	for {
		select {
		case <-ctx.Done():
			return
		case ev, ok := <-ch:
			if !ok {
				return
			}
			if err := writeEvent(conn, ev); err != nil {
				return
			}
		}
	}
}

func writeEvent(w io.Writer, ev interface{}) error {
	data, err := protocol.Encode(ev)
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	return err
}

// streamToWriter copies events from ch to a buffered writer. Not currently used
// by the library forwarder, but kept for symmetry with the reference example.
func streamToWriter(ctx context.Context, ch chan interface{}, w *bufio.Writer) {
	for {
		select {
		case <-ctx.Done():
			return
		case ev, ok := <-ch:
			if !ok {
				return
			}
			data, err := protocol.Encode(ev)
			if err != nil {
				continue
			}
			if _, err := w.Write(data); err != nil {
				return
			}
			w.Flush()
		}
	}
}
