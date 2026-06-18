package cmd

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"flashback-shell/pkg/config"
	"flashback-shell/pkg/log"
	"flashback-shell/pkg/shell"
)

// KillCmd handles the "kill" subcommand.
func KillCmd(cfg *config.Config, logger *log.Logger, args []string) int {
	fs := flag.NewFlagSet("kill", flag.ExitOnError)
	remainingArgs, err := config.ParseSubcommandFlags(fs, args)
	if err != nil {
		logger.Errorf("failed to parse flags: %v", err)
		return 1
	}

	logger.Infof("effective config: server_url=%s socket_dir=%s shell=%s buffer_size=%d device_id=%s capture_interval=%d disable_capture=%t",
		cfg.ServerURL, cfg.SocketDir, cfg.Shell, cfg.BufferSize, cfg.DeviceID, cfg.CaptureInterval, cfg.DisableCapture)

	if len(remainingArgs) < 1 {
		fmt.Fprintln(os.Stderr, "usage: flashback-shell kill <session-id>")
		return 1
	}

	sessionID := remainingArgs[0]
	name := fmt.Sprintf("flashback-%s", sessionID)
	socketPath := filepath.Join(cfg.SocketDir, name)

	sess := &shell.TmuxSession{
		Name:       name,
		SocketPath: socketPath,
	}

	if !sess.IsRunning() {
		logger.Errorf("session %s is not running", sessionID)
		return 1
	}

	logger.Infof("killing session %s", sessionID)
	if err := sess.Kill(); err != nil {
		logger.Errorf("failed to kill session: %v", err)
		return 1
	}

	fmt.Printf("killed session %s\n", sessionID)
	return 0
}
