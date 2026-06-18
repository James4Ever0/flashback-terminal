// logger connects to a forwarder socket and prints JSON events in a human-readable form.
package main

import (
	"bufio"
	"encoding/base64"
	"fmt"
	"net"
	"os"
	"strings"
	"time"

	"bash-forwarder-vt/pkg/event"
	"bash-forwarder-vt/pkg/protocol"
)

func usage() {
	fmt.Fprintf(os.Stderr, "Usage: %s <unix-socket-or-host:port>\n", os.Args[0])
}

func main() {
	if len(os.Args) != 2 {
		usage()
		os.Exit(1)
	}
	addr := os.Args[1]

	var conn net.Conn
	var err error
	if len(addr) > 0 && (addr[0] == '/' || addr[0] == '@') {
		conn, err = net.Dial("unix", addr)
	} else {
		conn, err = net.Dial("tcp", addr)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "connect: %v\n", err)
		os.Exit(1)
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(24 * time.Hour))

	br := bufio.NewReader(conn)
	for {
		line, err := br.ReadBytes('\n')
		if err != nil {
			fmt.Fprintf(os.Stderr, "read: %v\n", err)
			os.Exit(1)
		}
		ev, err := protocol.Decode(line)
		if err != nil {
			fmt.Fprintf(os.Stderr, "decode: %v\n", err)
			continue
		}
		if ev == nil {
			continue
		}
		printEvent(ev)
		switch ev.(type) {
		case *event.EofEvent:
			return
		}
	}
}

func printEvent(ev interface{}) {
	switch e := ev.(type) {
	case *event.InitEvent:
		fmt.Printf("[init] id=%d time=%d cols=%d rows=%d screen=%d bytes\n", e.ID, e.Time, e.Cols, e.Rows, len(e.Screen))
	case *event.OutputEvent:
		fmt.Printf("[output] id=%d time=%d data=%s\n", e.ID, e.Time, preview(e.Data))
	case *event.InputEvent:
		fmt.Printf("[input] id=%d time=%d data=%s\n", e.ID, e.Time, preview(e.Data))
	case *event.ResizeEvent:
		fmt.Printf("[resize] id=%d time=%d cols=%d rows=%d\n", e.ID, e.Time, e.Cols, e.Rows)
	case *event.MarkerEvent:
		fmt.Printf("[marker] id=%d time=%d label=%s\n", e.ID, e.Time, e.Label)
	case *event.ExitEvent:
		fmt.Printf("[exit] id=%d time=%d status=%d\n", e.ID, e.Time, e.Status)
	case *event.EofEvent:
		fmt.Printf("[eof] id=%d time=%d\n", e.ID, e.Time)
	}
}

func preview(b64 string) string {
	b, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return "[decode error]"
	}
	s := fmt.Sprintf("%q", string(b))
	if len(s) > 80 {
		return s[:77] + "..."
	}
	return strings.ReplaceAll(s, "\n", "\\n")
}
