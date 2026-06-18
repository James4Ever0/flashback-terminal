package main

import (
	"flag"
	"fmt"
	"os"

	"flashback-shell-pty/cmd"
	"flashback-shell-pty/pkg/config"
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

	verbose, configPath, logFile, noCapture, allowNested, allowNestedSet, subcommand, subArgs, err := parseArgs(os.Args[1:])
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
	if allowNestedSet {
		cfg.AllowNested = allowNested
		src.AllowNested = config.Source{Value: fmt.Sprintf("%t", allowNested), Origin: "cli"}
	}

	exitCode := 0
	switch subcommand {
	case "new":
		exitCode = cmd.NewCmd(cfg, logger, subArgs)
	case "check":
		exitCode = cmd.CheckCmd(cfg, src, logger, subArgs)
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", subcommand)
		printUsage()
		exitCode = 1
	}

	logger.Close()
	os.Exit(exitCode)
}

// parseArgs parses global flags and returns the subcommand with its remaining
// arguments. The returned noCapture and allowNested flags are applied by main
// as runtime overrides; allowNestedSet indicates whether --allow-nested was
// explicitly provided.
func parseArgs(osArgs []string) (verbose int, configPath, logFile string, noCapture, allowNested bool, allowNestedSet bool, command string, subArgs []string, err error) {
	globalFS := flag.NewFlagSet("flashback-shell-pty", flag.ContinueOnError)
	globalFS.Usage = func() {} // suppress default usage on error
	cp := globalFS.String("c", "", "path to config file")
	lf := globalFS.String("l", "", "path to log file")
	nc := globalFS.Bool("no-capture", false, "disable background capture for this session")
	an := globalFS.Bool("allow-nested", false, "allow starting a session inside an existing flashback-shell session")

	verbose, remaining := config.ExtractVerbose(osArgs)
	if perr := globalFS.Parse(remaining); perr != nil {
		err = perr
		return
	}

	globalFS.Visit(func(f *flag.Flag) {
		if f.Name == "allow-nested" {
			allowNestedSet = true
		}
	})

	args := globalFS.Args()
	if len(args) < 1 {
		err = fmt.Errorf("no command specified")
		return
	}

	return verbose, *cp, *lf, *nc, *an, allowNestedSet, args[0], args[1:], nil
}

func printUsage() {
	fmt.Println("flashback-shell-pty - Terminal session capture tool (PTY/VT)")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  flashback-shell-pty [global flags] <command> [command args]")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  new [args...]              Start a new shell inside a PTY")
	fmt.Println("  check                      Validate dependencies and show effective config values")
	fmt.Println()
	fmt.Println("Global flags:")
	fmt.Println("  -c <path>                Config file path (default: ~/.config/terminal-capture-shell-pty.yaml)")
	fmt.Println("  -v, -vv, -vvv            Verbosity: warn, info, debug")
	fmt.Println("  -l <path>                Log output file (default: ~/.flashback-shell-pty/log/flashback-shell-pty.log)")
	fmt.Println("  -l -                     Log to stderr")
	fmt.Println("  -l /dev/stdout           Log to stdout (may corrupt PTY attach output)")
	fmt.Println("  --no-capture             Disable background capture for this session")
	fmt.Println("  --allow-nested           Allow starting a session inside an existing flashback-shell session")
	fmt.Println()
	fmt.Println("Environment variables (override config file values):")
	fmt.Println("  FLASHBACK_SHELL_PTY_SERVER_URL")
	fmt.Println("  FLASHBACK_SHELL_PTY_SHELL")
	fmt.Println("  FLASHBACK_SHELL_PTY_BUFFER_SIZE")
	fmt.Println("  FLASHBACK_SHELL_PTY_BUFFER_MODE")
	fmt.Println("  FLASHBACK_SHELL_PTY_BUFFER_DIR")
	fmt.Println("  FLASHBACK_SHELL_PTY_DEVICE_ID")
	fmt.Println("  FLASHBACK_SHELL_PTY_CAPTURE_INTERVAL")
	fmt.Println("  FLASHBACK_SHELL_PTY_FIRST_CAPTURE_DELAY")
	fmt.Println("  FLASHBACK_SHELL_PTY_DISABLE_CAPTURE")
	fmt.Println("  FLASHBACK_SHELL_PTY_DIFF_ONLY")
	fmt.Println("  FLASHBACK_SHELL_PTY_DIFF_MODE")
	fmt.Println("  FLASHBACK_SHELL_PTY_TEXT_ONLY")
	fmt.Println("  FLASHBACK_SHELL_PTY_SCROLLBACK_LINES")
	fmt.Println("  FLASHBACK_SHELL_PTY_ALLOW_NESTED")
	fmt.Println()
	fmt.Println("Config file options (default: ~/.config/terminal-capture-shell-pty.yaml):")
	fmt.Println("  server_url, shell, buffer_size, buffer_mode, buffer_dir, device_id,")
	fmt.Println("  capture_interval, first_capture_delay, disable_capture, diff_only,")
	fmt.Println("  diff_mode, text_only, scrollback_lines, allow_nested")
	fmt.Println()
	fmt.Println("Config precedence: cli flags > env vars > config file > defaults")
}
