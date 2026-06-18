package cmd

import (
	"flag"
	"fmt"

	"flashback-shell-pty/pkg/config"
	"flashback-shell-pty/pkg/log"
	"flashback-shell-pty/pkg/session"
)

// ListCmd handles the "list" subcommand.
func ListCmd(cfg *config.Config, logger *log.Logger, args []string) int {
	fs := flag.NewFlagSet("list", flag.ExitOnError)
	_, err := config.ParseSubcommandFlags(fs, args)
	if err != nil {
		logger.Errorf("failed to parse flags: %v", err)
		return 1
	}

	logger.Infof("effective config: server_url=%s socket_dir=%s shell=%s buffer_size=%d device_id=%s capture_interval=%d disable_capture=%t",
		cfg.ServerURL, cfg.SocketDir, cfg.Shell, cfg.BufferSize, cfg.DeviceID, cfg.CaptureInterval, cfg.DisableCapture)

	sessions, err := session.DiscoverSessions(cfg.SocketDir)
	if err != nil {
		logger.Errorf("failed to list sessions: %v", err)
		return 1
	}

	if len(sessions) == 0 {
		logger.Infof("no flashback-shell-pty sessions found")
		return 0
	}

	fmt.Printf("%-20s %-8s %-8s %-10s %s\n", "ID", "COLS", "ROWS", "ATTACHED", "SOCKET")
	for _, s := range sessions {
		client := session.NewClient(s.SocketPath)
		status, err := client.Status()
		if err != nil {
			logger.Debugf("session %s status failed: %v", s.ID, err)
			_ = session.RemoveStaleSocket(s.SocketPath)
			continue
		}
		fmt.Printf("%-20s %-8d %-8d %-10t %s\n", status.SessionID, status.Cols, status.Rows, status.Attached, s.SocketPath)
	}
	return 0
}
