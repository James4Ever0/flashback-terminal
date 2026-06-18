// Package renderer applies a stream of forwarder events to a terminal.
package renderer

import (
	"bufio"
	"encoding/base64"
	"fmt"
	"io"
	"os"
	"syscall"

	"flashback-shell-tmux-renderer/internal/event"
	"flashback-shell-tmux-renderer/internal/protocol"
	"golang.org/x/sys/unix"
)

// RenderLoop reads JSON-RPC events from r and writes the reconstructed output
// to w. It returns when r is closed or an EOF event is received.
// If stdoutFD is a terminal, resize events also attempt TIOCSWINSZ in addition
// to the ANSI resize sequence.
func RenderLoop(r io.Reader, w io.Writer, stdoutFD int) error {
	br := bufio.NewReader(r)
	for {
		line, err := br.ReadBytes('\n')
		if err == io.EOF && len(line) == 0 {
			return nil
		}
		if err != nil {
			return err
		}
		ev, err := protocol.Decode(line)
		if err != nil {
			fmt.Fprintf(os.Stderr, "renderer: decode error: %v\n", err)
			continue
		}
		if ev == nil {
			continue
		}
		if err := applyEvent(w, stdoutFD, ev); err != nil {
			return err
		}
		switch ev.(type) {
		case *event.EofEvent:
			return nil
		}
	}
}

func applyEvent(w io.Writer, stdoutFD int, ev interface{}) error {
	switch e := ev.(type) {
	case *event.InitEvent:
		if e.Screen != "" {
			b, err := base64.StdEncoding.DecodeString(e.Screen)
			if err != nil {
				return err
			}
			if _, err := w.Write(b); err != nil {
				return err
			}
		}
		emitResize(w, stdoutFD, e.Rows, e.Cols)
		return nil
	case *event.OutputEvent:
		b, err := base64.StdEncoding.DecodeString(e.Data)
		if err != nil {
			return err
		}
		_, err = w.Write(b)
		return err
	case *event.InputEvent:
		return nil
	case *event.ResizeEvent:
		emitResize(w, stdoutFD, e.Rows, e.Cols)
		return nil
	case *event.MarkerEvent, *event.ExitEvent, *event.EofEvent:
		return nil
	}
	return nil
}

func emitResize(w io.Writer, stdoutFD int, rows, cols uint16) {
	// ANSI resize sequence.
	_, _ = fmt.Fprintf(w, "\x1b[8;%d;%dt", rows, cols)
	// Also try ioctl resize when we have a real terminal fd.
	if stdoutFD >= 0 {
		_ = unix.IoctlSetWinsize(stdoutFD, syscall.TIOCSWINSZ, &unix.Winsize{Row: rows, Col: cols})
	}
}

// DrainStdin consumes stdin until it is closed. This prevents accidental
// flow-control issues when the renderer is run inside tmux and the pane
// receives keystrokes.
func DrainStdin() {
	go func() {
		buf := make([]byte, 1024)
		for {
			n, err := os.Stdin.Read(buf)
			if n == 0 || err != nil {
				return
			}
		}
	}()
}
