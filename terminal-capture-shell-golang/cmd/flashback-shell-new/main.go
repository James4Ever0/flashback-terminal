package main

import (
	"fmt"
	"os"

	"flashback-shell/cmd"
	"flashback-shell/pkg/config"
)

var version = "dev"

func main() {
	// No config file, no CLI flags: configuration comes only from environment
	// variables (FLASHBACK_SHELL_*). All arguments are passed directly to the
	// spawned shell, including --help and -h.
	cfg, _, logger, err := config.LoadFromEnv(0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load config: %v\n", err)
		os.Exit(1)
	}

	exitCode := cmd.NewCmd(cfg, logger, os.Args[1:])

	logger.Close()
	os.Exit(exitCode)
}
