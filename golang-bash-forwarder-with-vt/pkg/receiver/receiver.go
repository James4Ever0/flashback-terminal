// Package receiver applies a stream of events to a terminal.
package receiver

import (
	"bufio"
	"encoding/base64"
	"fmt"
	"io"
	"os"

	"bash-forwarder-vt/pkg/event"
	"bash-forwarder-vt/pkg/protocol"
)

// RenderLoop reads JSON-RPC events from r and writes the reconstructed output
// to w. It returns when r is closed or an EOF event is received.
func RenderLoop(r io.Reader, w io.Writer) error {
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
			fmt.Fprintf(os.Stderr, "receiver: decode error: %v\n", err)
			continue
		}
		if ev == nil {
			continue
		}
		if err := applyEvent(w, ev); err != nil {
			return err
		}
		switch ev.(type) {
		case *event.EofEvent:
			return nil
		}
	}
}

func applyEvent(w io.Writer, ev interface{}) error {
	switch e := ev.(type) {
	case *event.InitEvent:
		// Decode the base64 screen snapshot and write it as a starting frame.
		if e.Screen != "" {
			b, err := base64.StdEncoding.DecodeString(e.Screen)
			if err != nil {
				return err
			}
			if _, err := w.Write(b); err != nil {
				return err
			}
		}
		// Also emit an ANSI resize sequence so the terminal size matches.
		_, err := fmt.Fprintf(w, "\x1b[8;%d;%dt", e.Rows, e.Cols)
		return err
	case *event.OutputEvent:
		b, err := base64.StdEncoding.DecodeString(e.Data)
		if err != nil {
			return err
		}
		_, err = w.Write(b)
		return err
	case *event.InputEvent:
		// By default we ignore input events; a logger may want to print them.
		return nil
	case *event.ResizeEvent:
		_, err := fmt.Fprintf(w, "\x1b[8;%d;%dt", e.Rows, e.Cols)
		return err
	case *event.MarkerEvent, *event.ExitEvent, *event.EofEvent:
		return nil
	}
	return nil
}
