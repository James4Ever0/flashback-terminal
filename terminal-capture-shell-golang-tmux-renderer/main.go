package main

import (
	"flag"
	"fmt"
	"os"

	"flashback-shell-tmux-renderer/internal/cmd"
	"flashback-shell-tmux-renderer/internal/config"
)

func main() {
	// Handle bare help first
	if len(os.Args) < 2 || os.Args[1] == "help" || os.Args[1] == "-h" || os.Args[1] == "--help" {
		printUsage()
		if len(os.Args) >= 2 && os.Args[1] == "help" {
			return
		}
		os.Exit(0)
	}

	verbose, configPath, logFile, noCapture, subcommand, subArgs, err := parseArgs(os.Args[1:])
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid arguments: %v\n", err)
		printUsage()
		os.Exit(1)
	}

	cfg, src, logger, err := config.LoadConfigAndLogger(configPath, logFile, verbose)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load config: %v\n", err)
		os.Exit(1)
	}

	// Runtime CLI override takes precedence over env/config defaults.
	if noCapture {
		cfg.DisableCapture = true
		src.DisableCapture = config.Source{Value: "true", Origin: "cli"}
	}

	exitCode := 0
	switch subcommand {
	case "new":
		exitCode = cmd.NewCmd(cfg, logger, subArgs)
	case "capture":
		exitCode = cmd.CaptureCmd(cfg, logger, subArgs)
	case "list":
		exitCode = cmd.ListCmd(cfg, logger, subArgs)
	case "kill":
		exitCode = cmd.KillCmd(cfg, logger, subArgs)
	case "check":
		exitCode = cmd.CheckCmd(cfg, src, logger, subArgs)
	case "__renderer":
		exitCode = cmd.RendererCmd(cfg, logger, subArgs)
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", subcommand)
		printUsage()
		exitCode = 1
	}

	logger.Close()
	os.Exit(exitCode)
}

// parseArgs parses global flags and returns the subcommand with its remaining
// arguments. The returned noCapture flag is applied by main as a runtime
// override.
func parseArgs(osArgs []string) (verbose int, configPath, logFile string, noCapture bool, command string, subArgs []string, err error) {
	globalFS := flag.NewFlagSet("flashback-shell", flag.ContinueOnError)
	globalFS.Usage = func() {} // suppress default usage on error
	cp := globalFS.String("c", "", "path to config file")
	lf := globalFS.String("l", "", "path to log file")
	nc := globalFS.Bool("no-capture", false, "disable background capture for this session")

	verbose, remaining := config.ExtractVerbose(osArgs)
	if perr := globalFS.Parse(remaining); perr != nil {
		err = perr
		return
	}

	args := globalFS.Args()
	if len(args) < 1 {
		err = fmt.Errorf("no command specified")
		return
	}

	return verbose, *cp, *lf, *nc, args[0], args[1:], nil
}

func printUsage() {
	fmt.Println("flashback-shell - Terminal session capture tool (socket-based forwarder + tmux renderer)")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  flashback-shell [global flags] <command> [command args]")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  new [args...]              Start a new shell (PTY forwarder + tmux renderer)")
	fmt.Println("  capture                    Capture all renderer tmux sessions and upload changes")
	fmt.Println("  list                       List managed sessions")
	fmt.Println("  kill <id>                  Kill a specific session")
	fmt.Println("  check                      Validate dependencies and show effective config values")
	fmt.Println()
	fmt.Println("Global flags:")
	fmt.Println("  -c <path>                Config file path (default: ~/.config/terminal-capture-shell.yaml)")
	fmt.Println("  -v, -vv, -vvv            Verbosity: warn, info, debug")
	fmt.Println("  -l <path>                Log output file (default: ~/.flashback-shell/log/flashback-shell.log)")
	fmt.Println("  -l -                     Log to stderr")
	fmt.Println("  -l /dev/stdout           Log to stdout (may corrupt terminal output)")
	fmt.Println("  --no-capture             Disable background capture for this session")
	fmt.Println()
	fmt.Println("Environment variables (override config file values):")
	fmt.Println("  FLASHBACK_SHELL_SERVER_URL          server_url")
	fmt.Println("  FLASHBACK_SHELL_SOCKET_DIR          socket_dir")
	fmt.Println("  FLASHBACK_SHELL_SHELL               shell")
	fmt.Println("  FLASHBACK_SHELL_BUFFER_SIZE         buffer_size")
	fmt.Println("  FLASHBACK_SHELL_DEVICE_ID           device_id")
	fmt.Println("  FLASHBACK_SHELL_CAPTURE_INTERVAL    capture_interval")
	fmt.Println("  FLASHBACK_SHELL_DISABLE_CAPTURE     disable_capture (true/false)")
	fmt.Println("  FLASHBACK_SHELL_CAPTURE_SCROLLBACK  capture_scrollback (true/false)")
	fmt.Println("  FLASHBACK_SHELL_DIFF_ONLY           diff_only (true/false)")
	fmt.Println("  FLASHBACK_SHELL_DIFF_MODE           diff_mode (suffix or index)")
	fmt.Println("  FLASHBACK_SHELL_TEXT_ONLY           text_only (true/false)")
	fmt.Println()
	fmt.Println("Config file options (default: ~/.config/terminal-capture-shell.yaml):")
	fmt.Println("  server_url, socket_dir, shell, buffer_size, device_id, capture_interval,")
	fmt.Println("  disable_capture, capture_scrollback, diff_only, diff_mode, text_only")
	fmt.Println()
	fmt.Println("Config precedence: cli flags > env vars > config file > defaults")
}
