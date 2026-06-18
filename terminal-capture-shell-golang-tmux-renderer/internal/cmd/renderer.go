package cmd

import (
	"fmt"
	"net"
	"os"
	"time"

	"flashback-shell-tmux-renderer/internal/config"
	"flashback-shell-tmux-renderer/internal/log"
	"flashback-shell-tmux-renderer/internal/renderer"
)

// RendererCmd handles the hidden "__renderer" subcommand. It is invoked by tmux
// to run the built-in renderer inside a tmux pane.
func RendererCmd(cfg *config.Config, logger *log.Logger, args []string) int {
	if len(args) < 1 {
		fmt.Fprintf(os.Stderr, "usage: flashback-shell __renderer <socket-path>\n")
		return 1
	}
	addr := args[0]

	renderer.DrainStdin()

	// Retry connection for a few seconds to tolerate the race between tmux
	// starting the renderer and the forwarder creating its socket.
	var conn net.Conn
	var err error
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if len(addr) > 0 && (addr[0] == '/' || addr[0] == '@') {
			conn, err = net.Dial("unix", addr)
		} else {
			conn, err = net.Dial("tcp", addr)
		}
		if err == nil {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if err != nil {
		logger.Errorf("connect to %s: %v", addr, err)
		return 1
	}
	defer conn.Close()

	// Set a long read timeout so a hung forwarder eventually releases us.
	_ = conn.SetDeadline(time.Now().Add(24 * time.Hour))

	logger.Debugf("renderer connected to %s", addr)
	if err := renderer.RenderLoop(conn, os.Stdout, int(os.Stdout.Fd())); err != nil {
		logger.Debugf("renderer loop ended: %v", err)
		return 0
	}
	return 0
}
