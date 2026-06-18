package cmd

import (
	"flag"
	"fmt"

	"flashback-shell-pty/pkg/config"
	"flashback-shell-pty/pkg/log"
	"flashback-shell-pty/pkg/session"
)

// KillCmd handles the "kill" subcommand.
func KillCmd(cfg *config.Config, logger *log.Logger, args []string) int {
	fs := flag.NewFlagSet("kill", flag.ExitOnError)
	_, err := config.ParseSubcommandFlags(fs, args)
	if err != nil {
		logger.Errorf("failed to parse flags: %v", err)
		return 1
	}

	logger.Infof("effective config: server_url=%s socket_dir=%s shell=%s buffer_size=%d device_id=%s capture_interval=%d disable_capture=%t",
		cfg.ServerURL, cfg.SocketDir, cfg.Shell, cfg.BufferSize, cfg.DeviceID, cfg.CaptureInterval, cfg.DisableCapture)

	if len(fs.Args()) == 0 {
		logger.Errorf("usage: kill <session-id>")
		return 1
	}

	sessionID := fs.Args()[0]
	socketPath := session.SocketPath(cfg.SocketDir, sessionID)
	client := session.NewClient(socketPath)
	if err := client.Kill(); err != nil {
		logger.Errorf("failed to kill session %s: %v", sessionID, err)
		return 1
	}

	fmt.Printf("killed session %s\n", sessionID)
	return 0
}
