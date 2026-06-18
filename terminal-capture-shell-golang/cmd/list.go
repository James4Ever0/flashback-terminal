package cmd

import (
	"flag"
	"fmt"

	"flashback-shell/pkg/config"
	"flashback-shell/pkg/log"
	"flashback-shell/pkg/shell"
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

	sessions, err := shell.DiscoverSessions(cfg.SocketDir)
	if err != nil {
		logger.Errorf("failed to list sessions: %v", err)
		return 1
	}

	if len(sessions) == 0 {
		logger.Infof("no flashback-shell tmux sessions found")
		return 0
	}

	fmt.Printf("%-36s %-30s %s\n", "ID", "NAME", "PANES")
	fmt.Println(string(make([]byte, 80)))
	for _, s := range sessions {
		sess := &shell.TmuxSession{
			Name:       s.Name,
			SocketPath: s.SocketPath,
		}
		panes, err := sess.ListPanes()
		paneCount := len(panes)
		if err != nil {
			paneCount = -1
		}
		fmt.Printf("%-36s %-30s %d\n", s.ID, s.Name, paneCount)
	}
	return 0
}
