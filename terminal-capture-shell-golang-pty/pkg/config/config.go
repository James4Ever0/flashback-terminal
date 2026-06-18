package config

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"flashback-shell-pty/pkg/log"
	"gopkg.in/yaml.v3"
)

// Config holds application configuration.
type Config struct {
	ServerURL       string `yaml:"server_url"`
	SocketDir       string `yaml:"socket_dir"`
	Shell           string `yaml:"shell"`
	BufferSize      int    `yaml:"buffer_size"`
	DeviceID        string `yaml:"device_id"`
	CaptureInterval  int    `yaml:"capture_interval"`
	FirstCaptureDelay int   `yaml:"first_capture_delay"`
	DisableCapture   bool   `yaml:"disable_capture"`
	DiffOnly        bool   `yaml:"diff_only"`
	DiffMode        string `yaml:"diff_mode"`
	TextOnly        bool   `yaml:"text_only"`
	ScrollbackLines int    `yaml:"scrollback_lines"`
	AllowNested     bool   `yaml:"allow_nested"`
}

// FileConfig is used for YAML unmarshaling to detect explicitly-set fields.
type FileConfig struct {
	ServerURL       string  `yaml:"server_url"`
	SocketDir       string  `yaml:"socket_dir"`
	Shell           string  `yaml:"shell"`
	BufferSize      int     `yaml:"buffer_size"`
	DeviceID        string  `yaml:"device_id"`
	CaptureInterval   *int    `yaml:"capture_interval"`
	FirstCaptureDelay *int    `yaml:"first_capture_delay"`
	DisableCapture    *bool   `yaml:"disable_capture"`
	DiffOnly        *bool   `yaml:"diff_only"`
	DiffMode        *string `yaml:"diff_mode"`
	TextOnly        *bool   `yaml:"text_only"`
	ScrollbackLines *int    `yaml:"scrollback_lines"`
	AllowNested     *bool   `yaml:"allow_nested"`
}

// Source tracks where a config value came from.
type Source struct {
	Value  string
	Origin string // "env", "config", "default", "missing"
}

// ConfigWithSource holds the effective config plus provenance for each field.
type ConfigWithSource struct {
	ServerURL       Source
	SocketDir       Source
	Shell           Source
	BufferSize      Source
	DeviceID        Source
	CaptureInterval  Source
	FirstCaptureDelay Source
	DisableCapture   Source
	DiffOnly        Source
	DiffMode        Source
	TextOnly        Source
	ScrollbackLines Source
	AllowNested     Source
}

// DefaultConfigPath returns the default YAML config file path.
func DefaultConfigPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "terminal-capture-shell-pty.yaml")
}

// Default returns a Config populated from the default config path with env overrides.
func Default() (*Config, *ConfigWithSource, error) {
	path := DefaultConfigPath()
	_, _ = EnsureDefaultConfig()
	return Load(path)
}

// Load reads config from a YAML file, applies env overrides, and returns effective values plus sources.
func Load(path string) (*Config, *ConfigWithSource, error) {
	cfg, src := defaultConfig()

	if path != "" {
		if data, err := os.ReadFile(path); err == nil {
			var fileCfg FileConfig
			if yerr := yaml.Unmarshal(data, &fileCfg); yerr != nil {
				return nil, nil, fmt.Errorf("parse config %s: %w", path, yerr)
			}
			applyYAML(cfg, src, &fileCfg)
		}
	}

	applyEnvOverrides(cfg, src)
	cfg.SocketDir = expandPath(cfg.SocketDir)

	return cfg, src, nil
}

// EnsureDefaultConfig creates the config directory and a template config file if missing.
func EnsureDefaultConfig() (string, error) {
	path := DefaultConfigPath()
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return path, fmt.Errorf("create config dir %s: %w", dir, err)
	}
	if _, err := os.Stat(path); os.IsNotExist(err) {
		if werr := WriteTemplate(path); werr != nil {
			return path, werr
		}
	}
	return path, nil
}

// WriteTemplate writes the default YAML template to the given path.
func WriteTemplate(path string) error {
	template := `# terminal-capture-shell-pty configuration
# Environment variables override values in this file.
# Available env vars (names match the config file keys):
#   FLASHBACK_SHELL_PTY_SERVER_URL
#   FLASHBACK_SHELL_PTY_SOCKET_DIR
#   FLASHBACK_SHELL_PTY_SHELL
#   FLASHBACK_SHELL_PTY_BUFFER_SIZE
#   FLASHBACK_SHELL_PTY_DEVICE_ID
#   FLASHBACK_SHELL_PTY_CAPTURE_INTERVAL
#   FLASHBACK_SHELL_PTY_FIRST_CAPTURE_DELAY
#   FLASHBACK_SHELL_PTY_DISABLE_CAPTURE
#   FLASHBACK_SHELL_PTY_DIFF_ONLY
#   FLASHBACK_SHELL_PTY_DIFF_MODE
#   FLASHBACK_SHELL_PTY_TEXT_ONLY
#   FLASHBACK_SHELL_PTY_SCROLLBACK_LINES
#   FLASHBACK_SHELL_PTY_ALLOW_NESTED

# Remote server URL for uploading captures (empty = local only)
server_url: ""

# Directory where session socket files are stored
socket_dir: "~/.flashback-shell-pty/sockets"

# Shell binary inside sessions (empty = $SHELL or /bin/bash)
shell: ""

# Maximum buffered capture batches kept for retry
buffer_size: 100

# Device identifier sent with uploads (empty = hostname)
device_id: ""

# Seconds between background captures for 'new' command (0 = disable)
capture_interval: 30

# Seconds to wait before the first background capture after 'new' starts.
first_capture_delay: 5

# Disable background capture entirely for 'new' command
disable_capture: false

# Only capture lines that newly appeared since the previous capture.
# First capture always returns the full buffer/screen.
diff_only: false

# Diff algorithm used when diff_only is true.
#   "suffix" (default) - find the longest overlapping suffix/prefix and return
#                      the trailing new lines. Best for ring-like scrollback.
#   "index"            - align previous and current buffers by index, trim or
#                      pad the top, and return only changed lines. Best for
#                      in-place screen updates.
diff_mode: suffix

# Send only plain text captures; omit ANSI escape codes from uploads.
text_only: false

# Maximum scrollback lines kept by the VT emulator.
scrollback_lines: 1000

# Allow starting a flashback-shell session inside another flashback-shell
# session. The shell inside a session sets FLASHBACK_SHELL=1; by default
# launching ` + "`new`" + ` while that variable is present is refused.
allow_nested: false
`
	if err := os.WriteFile(path, []byte(template), 0644); err != nil {
		return fmt.Errorf("write config template %s: %w", path, err)
	}
	return nil
}

// LoadConfigAndLogger loads the effective config and creates a logger from the given paths and verbosity.
func LoadConfigAndLogger(configPath, logFile string, verbose int) (*Config, *ConfigWithSource, *log.Logger, error) {
	cfgPath := configPath
	if cfgPath == "" {
		cfgPath, _ = EnsureDefaultConfig()
	}

	cfg, src, err := Load(cfgPath)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load config: %w", err)
	}

	out, logPath, err := OpenLogWriter(logFile)
	if err != nil {
		out = os.Stderr
		logPath = "stderr"
	}

	logger := log.New(verbose, out)
	if err != nil {
		logger.Warnf("cannot open log file %s, logging to stderr: %v", logPath, err)
	}
	logger.Debugf("using config file: %s", cfgPath)
	if logPath != "" {
		logger.Debugf("using log destination: %s", logPath)
	}

	return cfg, src, logger, nil
}

// OpenLogWriter opens the requested log destination. Use "" for the default log file.
func OpenLogWriter(logFile string) (io.Writer, string, error) {
	switch logFile {
	case "-":
		return os.Stderr, "stderr", nil
	case "/dev/stderr":
		return os.Stderr, "stderr", nil
	case "/dev/stdout":
		return os.Stdout, "stdout", nil
	}

	if logFile != "" {
		f, err := os.OpenFile(logFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			return nil, logFile, err
		}
		return f, logFile, nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return nil, "", fmt.Errorf("cannot determine home dir: %w", err)
	}
	logDir := filepath.Join(home, ".flashback-shell-pty", "log")
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return nil, logDir, fmt.Errorf("create log dir: %w", err)
	}
	logPath := filepath.Join(logDir, "flashback-shell-pty.log")
	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, logPath, err
	}
	return f, logPath, nil
}

// ParseSubcommandFlags parses only subcommand-specific flags and returns remaining positional args.
func ParseSubcommandFlags(fs interface {
	Parse(arguments []string) error
	Args() []string
}, args []string) ([]string, error) {
	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	return fs.Args(), nil
}

func defaultConfig() (*Config, *ConfigWithSource) {
	home, _ := os.UserHomeDir()
	cfg := &Config{
		SocketDir:         filepath.Join(home, ".flashback-shell-pty", "sockets"),
		BufferSize:        100,
		CaptureInterval:   30,
		FirstCaptureDelay: 5,
		DiffOnly:          false,
		DiffMode:          "suffix",
		TextOnly:          false,
		ScrollbackLines:   1000,
		AllowNested:       false,
	}
	src := &ConfigWithSource{
		ServerURL:         Source{Value: cfg.ServerURL, Origin: "missing"},
		SocketDir:         Source{Value: cfg.SocketDir, Origin: "default"},
		Shell:             Source{Value: cfg.Shell, Origin: "missing"},
		BufferSize:        Source{Value: fmt.Sprintf("%d", cfg.BufferSize), Origin: "default"},
		DeviceID:          Source{Value: cfg.DeviceID, Origin: "missing"},
		CaptureInterval:   Source{Value: fmt.Sprintf("%d", cfg.CaptureInterval), Origin: "default"},
		FirstCaptureDelay: Source{Value: fmt.Sprintf("%d", cfg.FirstCaptureDelay), Origin: "default"},
		DisableCapture:    Source{Value: "false", Origin: "default"},
		DiffOnly:          Source{Value: "false", Origin: "default"},
		DiffMode:          Source{Value: "suffix", Origin: "default"},
		TextOnly:          Source{Value: "false", Origin: "default"},
		ScrollbackLines:   Source{Value: fmt.Sprintf("%d", cfg.ScrollbackLines), Origin: "default"},
		AllowNested:       Source{Value: "false", Origin: "default"},
	}
	return cfg, src
}

func applyYAML(cfg *Config, src *ConfigWithSource, fileCfg *FileConfig) {
	if fileCfg.ServerURL != "" {
		cfg.ServerURL = fileCfg.ServerURL
		src.ServerURL = Source{Value: cfg.ServerURL, Origin: "config"}
	}
	if fileCfg.SocketDir != "" {
		cfg.SocketDir = fileCfg.SocketDir
		src.SocketDir = Source{Value: cfg.SocketDir, Origin: "config"}
	}
	if fileCfg.Shell != "" {
		cfg.Shell = fileCfg.Shell
		src.Shell = Source{Value: cfg.Shell, Origin: "config"}
	}
	if fileCfg.BufferSize != 0 {
		cfg.BufferSize = fileCfg.BufferSize
		src.BufferSize = Source{Value: fmt.Sprintf("%d", cfg.BufferSize), Origin: "config"}
	}
	if fileCfg.DeviceID != "" {
		cfg.DeviceID = fileCfg.DeviceID
		src.DeviceID = Source{Value: cfg.DeviceID, Origin: "config"}
	}
	if fileCfg.CaptureInterval != nil {
		cfg.CaptureInterval = *fileCfg.CaptureInterval
		src.CaptureInterval = Source{Value: fmt.Sprintf("%d", cfg.CaptureInterval), Origin: "config"}
	}
	if fileCfg.FirstCaptureDelay != nil {
		cfg.FirstCaptureDelay = *fileCfg.FirstCaptureDelay
		src.FirstCaptureDelay = Source{Value: fmt.Sprintf("%d", cfg.FirstCaptureDelay), Origin: "config"}
	}
	if fileCfg.DisableCapture != nil {
		cfg.DisableCapture = *fileCfg.DisableCapture
		src.DisableCapture = Source{Value: fmt.Sprintf("%t", cfg.DisableCapture), Origin: "config"}
	}
	if fileCfg.DiffOnly != nil {
		cfg.DiffOnly = *fileCfg.DiffOnly
		src.DiffOnly = Source{Value: fmt.Sprintf("%t", cfg.DiffOnly), Origin: "config"}
	}
	if fileCfg.DiffMode != nil && *fileCfg.DiffMode != "" {
		cfg.DiffMode = *fileCfg.DiffMode
		src.DiffMode = Source{Value: cfg.DiffMode, Origin: "config"}
	}
	if fileCfg.TextOnly != nil {
		cfg.TextOnly = *fileCfg.TextOnly
		src.TextOnly = Source{Value: fmt.Sprintf("%t", cfg.TextOnly), Origin: "config"}
	}
	if fileCfg.ScrollbackLines != nil {
		cfg.ScrollbackLines = *fileCfg.ScrollbackLines
		src.ScrollbackLines = Source{Value: fmt.Sprintf("%d", cfg.ScrollbackLines), Origin: "config"}
	}
	if fileCfg.AllowNested != nil {
		cfg.AllowNested = *fileCfg.AllowNested
		src.AllowNested = Source{Value: fmt.Sprintf("%t", cfg.AllowNested), Origin: "config"}
	}
}

func applyEnvOverrides(cfg *Config, src *ConfigWithSource) {
	if v := os.Getenv("FLASHBACK_SHELL_PTY_SERVER_URL"); v != "" {
		cfg.ServerURL = v
		src.ServerURL = Source{Value: v, Origin: "env"}
	}
	if v := os.Getenv("FLASHBACK_SHELL_PTY_SOCKET_DIR"); v != "" {
		cfg.SocketDir = v
		src.SocketDir = Source{Value: v, Origin: "env"}
	}
	if v := os.Getenv("FLASHBACK_SHELL_PTY_SHELL"); v != "" {
		cfg.Shell = v
		src.Shell = Source{Value: v, Origin: "env"}
	}
	if v := os.Getenv("FLASHBACK_SHELL_PTY_BUFFER_SIZE"); v != "" {
		if n := parseInt(v); n > 0 {
			cfg.BufferSize = n
			src.BufferSize = Source{Value: v, Origin: "env"}
		}
	}
	if v := os.Getenv("FLASHBACK_SHELL_PTY_DEVICE_ID"); v != "" {
		cfg.DeviceID = v
		src.DeviceID = Source{Value: v, Origin: "env"}
	}
	if v := os.Getenv("FLASHBACK_SHELL_PTY_CAPTURE_INTERVAL"); v != "" {
		if n := parseInt(v); n >= 0 {
			cfg.CaptureInterval = n
			src.CaptureInterval = Source{Value: v, Origin: "env"}
		}
	}
	if v := os.Getenv("FLASHBACK_SHELL_PTY_FIRST_CAPTURE_DELAY"); v != "" {
		if n := parseInt(v); n >= 0 {
			cfg.FirstCaptureDelay = n
			src.FirstCaptureDelay = Source{Value: v, Origin: "env"}
		}
	}
	if v := os.Getenv("FLASHBACK_SHELL_PTY_DISABLE_CAPTURE"); v != "" {
		cfg.DisableCapture = parseBool(v)
		src.DisableCapture = Source{Value: v, Origin: "env"}
	}
	if v := os.Getenv("FLASHBACK_SHELL_PTY_DIFF_ONLY"); v != "" {
		cfg.DiffOnly = parseBool(v)
		src.DiffOnly = Source{Value: v, Origin: "env"}
	}
	if v := os.Getenv("FLASHBACK_SHELL_PTY_DIFF_MODE"); v != "" {
		cfg.DiffMode = v
		src.DiffMode = Source{Value: v, Origin: "env"}
	}
	if v := os.Getenv("FLASHBACK_SHELL_PTY_TEXT_ONLY"); v != "" {
		cfg.TextOnly = parseBool(v)
		src.TextOnly = Source{Value: v, Origin: "env"}
	}
	if v := os.Getenv("FLASHBACK_SHELL_PTY_SCROLLBACK_LINES"); v != "" {
		if n := parseInt(v); n >= 0 {
			cfg.ScrollbackLines = n
			src.ScrollbackLines = Source{Value: v, Origin: "env"}
		}
	}
	if v := os.Getenv("FLASHBACK_SHELL_PTY_ALLOW_NESTED"); v != "" {
		cfg.AllowNested = parseBool(v)
		src.AllowNested = Source{Value: v, Origin: "env"}
	}
}

func parseInt(s string) int {
	var n int
	for _, c := range s {
		if c < '0' || c > '9' {
			return -1
		}
		n = n*10 + int(c-'0')
	}
	return n
}

func parseBool(s string) bool {
	s = strings.ToLower(strings.TrimSpace(s))
	return s == "true" || s == "1" || s == "yes" || s == "on"
}

func expandPath(s string) string {
	if strings.HasPrefix(s, "~/") {
		home, err := os.UserHomeDir()
		if err == nil {
			return filepath.Join(home, s[2:])
		}
	}
	return s
}

// ExtractVerbose scans args for -v, -vv, -vvv and returns the count plus args with those removed.
func ExtractVerbose(args []string) (int, []string) {
	count := 0
	var rest []string
	for _, arg := range args {
		switch arg {
		case "-v":
			count++
		case "-vv":
			count += 2
		case "-vvv":
			count += 3
		default:
			rest = append(rest, arg)
		}
	}
	return count, rest
}
