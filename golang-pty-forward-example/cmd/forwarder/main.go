// forwarder wraps a program in a PTY, mirrors the local terminal, and forwards
// typed JSON-RPC events to a Unix socket listener.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"pty-forward-example/pkg/event"
	"pty-forward-example/pkg/protocol"
	"pty-forward-example/pkg/ptywrap"
	"pty-forward-example/pkg/rawmode"
)

var (
	listenAddr = flag.String("listen", "", "Unix socket or host:port to forward events to")
	logFile    = flag.String("log", "", "path to write JSON event log (- for stdout)")
	headless   = flag.Bool("headless", false, "do not attach the local terminal (non-interactive)")
	noCapture  = flag.Bool("no-capture", false, "do not capture input events")
	verbose    = flag.Bool("v", false, "enable verbose debug logging to stderr")
)

// debugf logs to stderr only when -v is set.
func debugf(format string, args ...interface{}) {
	if *verbose {
		log.Printf(format, args...)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, "Usage: %s [flags] <command> [args...]\n\nFlags:\n", filepath.Base(os.Args[0]))
	flag.PrintDefaults()
}

// setEnv updates or appends a key=value pair in a process environment slice.
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

func main() {
	flag.Usage = usage
	flag.Parse()
	if flag.NArg() == 0 {
		flag.Usage()
		os.Exit(1)
	}

	cmdArgs := flag.Args()
	cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)

	// Build the child's environment with the correct terminal size before starting.
	// term.GetSize returns (width, height) = (cols, rows).
	env := os.Environ()
	if !*headless && rawmode.IsTerminal(int(os.Stdin.Fd())) {
		cols, rows, err := rawmode.Size(int(os.Stdin.Fd()))
		debugf("[env] rawmode.Size(fd=%d) = cols=%d rows=%d err=%v", int(os.Stdin.Fd()), cols, rows, err)
		if err == nil && cols > 0 && rows > 0 {
			env = setEnv(env, "COLUMNS", fmt.Sprintf("%d", cols))
			env = setEnv(env, "LINES", fmt.Sprintf("%d", rows))
			debugf("[env] set COLUMNS=%d LINES=%d", cols, rows)
		}
	}
	if os.Getenv("TERM") == "" {
		env = setEnv(env, "TERM", "xterm-256color")
		debugf("[env] set TERM=xterm-256color")
	}
	cmd.Env = env
	for _, e := range env {
		if len(e) >= 8 && (e[:8] == "COLUMNS=" || e[:6] == "LINES=" || e[:5] == "TERM=") {
			debugf("[env] cmd.Env entry: %s", e)
		}
	}

	ptmx, err := ptywrap.Start(cmd)
	if err != nil {
		log.Fatalf("failed to start pty: %v", err)
	}
	defer ptmx.Close()
	childPid := cmd.Process.Pid

	bus := event.NewBus()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	stdinDone := make(chan struct{})

	var wg sync.WaitGroup

	// Set initial size from the local terminal.
	if !*headless && rawmode.IsTerminal(int(os.Stdin.Fd())) {
		cols, rows, err := rawmode.Size(int(os.Stdin.Fd()))
		debugf("[init resize] rawmode.Size(fd=%d) = cols=%d rows=%d err=%v", int(os.Stdin.Fd()), cols, rows, err)
		if err == nil && cols > 0 && rows > 0 {
			debugf("[init resize] ptywrap.Resize(ptmx, rows=%d, cols=%d)", rows, cols)
			_ = ptywrap.Resize(ptmx, uint16(rows), uint16(cols))
			debugf("[init resize] bus.SetScreen(cols=%d, rows=%d)", cols, rows)
			bus.SetScreen(uint16(cols), uint16(rows), nil)
			id, t := bus.NextID()
			debugf("[init resize] publish ResizeEvent{Cols=%d, Rows=%d}", cols, rows)
			bus.Publish(&event.ResizeEvent{Event: event.Event{ID: id, Time: t}, Cols: uint16(cols), Rows: uint16(rows)})
		}
	}

	// Listener / file logger setup.
	var listener net.Listener
	if *listenAddr != "" {
		if isUnixSocket(*listenAddr) {
			_ = os.Remove(*listenAddr)
			l, err := net.Listen("unix", *listenAddr)
			if err != nil {
				log.Fatalf("listen %s: %v", *listenAddr, err)
			}
			listener = l
		} else {
			l, err := net.Listen("tcp", *listenAddr)
			if err != nil {
				log.Fatalf("listen %s: %v", *listenAddr, err)
			}
			listener = l
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			serveListener(ctx, listener, bus)
		}()
	}

	var fileLogger *bufio.Writer
	var fileLoggerCloser io.Closer
	if *logFile != "" {
		if *logFile == "-" {
			// Write event log to stdout; do not close stdout.
			fileLogger = bufio.NewWriter(os.Stdout)
		} else {
			f, err := os.Create(*logFile)
			if err != nil {
				log.Fatalf("log file: %v", err)
			}
			fileLoggerCloser = f
			fileLogger = bufio.NewWriter(f)
		}
		if _, init, ch := bus.Subscribe(); ch != nil {
			// Subscribe so file logger receives events too.
			// Write the init event first so the log contains the starting state.
			if data, err := protocol.Encode(init); err == nil {
				if _, err := fileLogger.Write(data); err == nil {
					fileLogger.Flush()
				}
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				streamToWriter(ctx, ch, fileLogger)
			}()
		}
	}

	// Signal handling for clean shutdown.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	var restore func() error
	if !*headless && rawmode.IsTerminal(int(os.Stdin.Fd())) {
		var err error
		restore, _, err = rawmode.MakeRaw(int(os.Stdin.Fd()))
		if err != nil {
			log.Fatalf("make raw: %v", err)
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
				if !*headless && rawmode.IsTerminal(int(os.Stdin.Fd())) {
					cols, rows, err := rawmode.Size(int(os.Stdin.Fd()))
					debugf("[sigwinch] rawmode.Size(fd=%d) = cols=%d rows=%d err=%v", int(os.Stdin.Fd()), cols, rows, err)
					if err == nil && cols > 0 && rows > 0 {
						debugf("[sigwinch] ptywrap.Resize(ptmx, rows=%d, cols=%d)", rows, cols)
						_ = ptywrap.Resize(ptmx, uint16(rows), uint16(cols))
						id, t := bus.NextID()
						debugf("[sigwinch] publish ResizeEvent{Cols=%d, Rows=%d}", cols, rows)
						bus.Publish(&event.ResizeEvent{Event: event.Event{ID: id, Time: t}, Cols: uint16(cols), Rows: uint16(rows)})
						// Forward SIGWINCH so the child's process group re-reads its terminal size.
						if childPid > 0 {
							debugf("[sigwinch] forward SIGWINCH to child process group -%d", childPid)
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
				if !*noCapture {
					id, t := bus.NextID()
					bus.Publish(&event.InputEvent{Event: event.Event{ID: id, Time: t}, Data: protocol.EncodeBytes(buf[:n])})
				}
			}
			if err != nil {
				return
			}
		}
	}()

	// Forward PTY -> stdout and bus.
	wg.Add(1)
	go func() {
		defer wg.Done()
		buf := make([]byte, 4096)
		for {
			n, err := ptmx.Read(buf)
			if n > 0 {
				if !*headless {
					_, _ = os.Stdout.Write(buf[:n])
				}
				id, t := bus.NextID()
				bus.Publish(&event.OutputEvent{Event: event.Event{ID: id, Time: t}, Data: protocol.EncodeBytes(buf[:n])})
			}
			if err != nil {
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

	// cleanup restores the local terminal. We do not wait on the stdin goroutine,
	// but closing fd 0 is kept as a best-effort attempt to wake it.
	cleanup := func() {
		if restore != nil {
			_ = restore()
			restore = nil
		}
		_ = syscall.Close(0)
	}

	select {
	case sig := <-sigCh:
		debugf("received signal %s, forwarding to child process group", sig)
		if childPid > 0 {
			if s, ok := sig.(syscall.Signal); ok {
				_ = syscall.Kill(-childPid, s)
			}
		}
		select {
		case <-waitDone:
			cleanup()
		case <-time.After(5 * time.Second):
			debugf("child did not exit gracefully, sending SIGKILL")
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
	// Flush and close the file logger before waiting for goroutines so the log
	// file contains all events including exit/eof.
	if fileLogger != nil {
		fileLogger.Flush()
	}
	if fileLoggerCloser != nil {
		fileLoggerCloser.Close()
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
		debugf("timeout waiting for goroutines to finish")
	}

	os.Exit(status)
}

func isUnixSocket(addr string) bool {
	return filepath.IsAbs(addr) || (len(addr) > 0 && addr[0] == '@')
}

func serveListener(ctx context.Context, listener net.Listener, bus *event.Bus) {
	for {
		conn, err := listener.Accept()
		if err != nil {
			select {
			case <-ctx.Done():
				return
			default:
				debugf("accept error: %v", err)
				continue
			}
		}
		go handleConn(ctx, conn, bus)
	}
}

func handleConn(ctx context.Context, conn net.Conn, bus *event.Bus) {
	defer conn.Close()
	subID, init, ch := bus.Subscribe()
	defer bus.Unsubscribe(subID)

	if err := writeEvent(conn, init); err != nil {
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
