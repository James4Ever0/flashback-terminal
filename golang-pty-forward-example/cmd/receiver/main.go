// receiver connects to a forwarder socket and replays terminal events to stdout.
package main

import (
	"fmt"
	"net"
	"os"
	"time"

	"pty-forward-example/pkg/receiver"
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

	// Set a read timeout so a hung forwarder eventually releases us.
	_ = conn.SetDeadline(time.Now().Add(24 * time.Hour))

	if err := receiver.RenderLoop(conn, os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "render: %v\n", err)
		os.Exit(1)
	}
}
