package shell

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// SessionInfo holds metadata about a tmux session.
type SessionInfo struct {
	ID         string
	Name       string
	SocketPath string
	PaneCount  int
}

// PaneInfo holds metadata about a tmux pane.
type PaneInfo struct {
	SessionID string
	PaneID    string
	Target    string // e.g. "session:0.0"
}

// TmuxSession manages a single tmux session lifecycle.
type TmuxSession struct {
	SessionID         string
	Name              string
	SocketPath        string
	SocketDir         string
	CWD               string
	Shell             string
	Args              []string
	CaptureScrollback bool
	AllowNestedTmux   bool
}

// NewTmuxSession creates a new tmux session descriptor.
func NewTmuxSession(sessionID, socketDir, cwd, shell string, args []string) *TmuxSession {
	name := fmt.Sprintf("flashback-%s", sessionID)
	return &TmuxSession{
		SessionID:  sessionID,
		Name:       name,
		SocketPath: filepath.Join(socketDir, name),
		SocketDir:  socketDir,
		CWD:        cwd,
		Shell:      shell,
		Args:       args,
	}
}

// Start creates the tmux session detached with kiosk mode settings.
func (s *TmuxSession) Start() error {
	if err := os.MkdirAll(s.SocketDir, 0755); err != nil {
		return fmt.Errorf("create socket dir: %w", err)
	}

	shell := s.Shell
	if shell == "" {
		shell = os.Getenv("SHELL")
		if shell == "" {
			shell = "/bin/bash"
		}
	}

	// Write a minimal tmux config so global options (especially default-terminal)
	// are applied before the first session is created.
	confPath := filepath.Join(s.SocketDir, "tmux.conf")
	conf := `# flashback-shell tmux kiosk configuration
set -g status off
set -g mouse off
set -g default-terminal "xterm-256color"
set -g default-command ""
`
	if err := os.WriteFile(confPath, []byte(conf), 0644); err != nil {
		return fmt.Errorf("write tmux config: %w", err)
	}

	// Build shell invocation
	envPath, _ := exec.LookPath("env")
	shellArgs := strings.Join(s.Args, " ")
	var startShellCmd string
	if s.AllowNestedTmux && envPath != "" {
		// Run the shell under env -u so the pane process starts without tmux
		// variables. This makes flashback-shell usable inside an existing tmux
		// session. env is used instead of shell-specific unset syntax.
		envPrefix := fmt.Sprintf("%s -u TMUX -u TMUX_PANE -u TMUX_WINDOW -u TMUX_SESSION -u TERM_PROGRAM", envPath)
		if shellArgs != "" {
			startShellCmd = fmt.Sprintf("cd %s && exec %s %s %s", s.CWD, envPrefix, shell, shellArgs)
		} else {
			startShellCmd = fmt.Sprintf("cd %s && exec %s %s", s.CWD, envPrefix, shell)
		}
	} else {
		if shellArgs != "" {
			startShellCmd = fmt.Sprintf("cd %s && exec %s %s", s.CWD, shell, shellArgs)
		} else {
			startShellCmd = fmt.Sprintf("cd %s && exec %s", s.CWD, shell)
		}
	}

	// Create detached session, loading the kiosk config.
	cmd := exec.Command("tmux",
		"-S", s.SocketPath,
		"-f", confPath,
		"new-session", "-d",
		"-s", s.Name,
		"-n", "main",
		"-e", "TERM=xterm-256color",
		"-e", "COLORTERM=truecolor",
		"-e", "FLASHBACK_SHELL_STARTED=1",
		startShellCmd,
	)
	cmd.Env = s.env()
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("tmux new-session: %w\n%s", err, string(out))
	}

	// Disable all key bindings after the session is created.
	c := exec.Command("tmux", "-S", s.SocketPath, "unbind-key", "-a")
	c.Env = s.env()
	c.CombinedOutput() // best-effort

	return nil
}

// Attach runs tmux attach as a subprocess with inherited stdio.
// Blocks until the user detaches or the session ends.
func (s *TmuxSession) Attach() error {
	cmd := exec.Command("tmux", "-S", s.SocketPath, "attach", "-t", s.Name)
	cmd.Env = s.env()
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// Kill terminates the tmux session and removes the socket file.
func (s *TmuxSession) Kill() error {
	cmd := exec.Command("tmux", "-S", s.SocketPath, "kill-session", "-t", s.Name)
	cmd.Env = s.env()
	cmd.CombinedOutput() // ignore errors; session may already be dead
	os.Remove(s.SocketPath)
	return nil
}

// IsRunning checks if the tmux session is still alive.
func (s *TmuxSession) IsRunning() bool {
	cmd := exec.Command("tmux", "-S", s.SocketPath, "has-session", "-t", s.Name)
	cmd.Env = s.env()
	if cmd.Run() != nil {
		return false
	}
	_, err := os.Stat(s.SocketPath)
	return err == nil
}

// CapturePane captures the content of a specific pane.
func (s *TmuxSession) CapturePane(paneTarget string, withANSI bool) (string, error) {
	args := []string{"-S", s.SocketPath, "capture-pane", "-p", "-J", "-t", paneTarget}
	if s.CaptureScrollback {
		args = append(args, "-S", "-")
	}
	if withANSI {
		args = append(args, "-e")
	}
	cmd := exec.Command("tmux", args...)
	cmd.Env = s.env()
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("capture-pane: %w\n%s", err, string(out))
	}
	return string(out), nil
}

// ListPanes returns all pane targets for this session.
// Targets are formatted as "session:window.pane" so they can be used directly
// with capture-pane and other tmux commands.
func (s *TmuxSession) ListPanes() ([]string, error) {
	cmd := exec.Command("tmux", "-S", s.SocketPath, "list-panes", "-t", s.Name, "-F", "#{window_index}.#{pane_index}")
	cmd.Env = s.env()
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("list-panes: %w\n%s", err, string(out))
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	var panes []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			panes = append(panes, fmt.Sprintf("%s:%s", s.Name, line))
		}
	}
	return panes, nil
}

func (s *TmuxSession) env() []string {
	var env []string
	for _, e := range os.Environ() {
		if s.AllowNestedTmux {
			if strings.HasPrefix(e, "TMUX=") ||
				strings.HasPrefix(e, "TMUX_PANE=") ||
				strings.HasPrefix(e, "TMUX_WINDOW=") ||
				strings.HasPrefix(e, "TMUX_SESSION=") ||
				strings.HasPrefix(e, "TERM_PROGRAM=") {
				continue
			}
		}
		// Drop outer TERM/COLORTERM so we can force 256/truecolor support
		if strings.HasPrefix(e, "TERM=") || strings.HasPrefix(e, "COLORTERM=") {
			continue
		}
		env = append(env, e)
	}
	env = append(env, "TERM=xterm-256color", "COLORTERM=truecolor")
	return env
}

// DiscoverSessions scans the socket directory for flashback-managed sessions.
// Dead sessions (socket file exists but tmux has no matching session) are
// removed and omitted from the result.
func DiscoverSessions(socketDir string) ([]SessionInfo, error) {
	entries, err := os.ReadDir(socketDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var sessions []SessionInfo
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || strings.HasSuffix(name, ".conf") {
			continue
		}
		if !strings.HasPrefix(name, "flashback-") {
			continue
		}
		id := strings.TrimPrefix(name, "flashback-")
		socketPath := filepath.Join(socketDir, name)

		// Skip stale sockets that no longer correspond to a live tmux session.
		sess := &TmuxSession{Name: name, SocketPath: socketPath}
		if !sess.IsRunning() {
			os.Remove(socketPath)
			continue
		}

		sessions = append(sessions, SessionInfo{
			ID:         id,
			Name:       name,
			SocketPath: socketPath,
		})
	}
	return sessions, nil
}
