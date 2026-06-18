// Package tmux manages the renderer tmux sessions used by flashback-shell.
package tmux

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// SessionInfo holds metadata about a managed renderer tmux session.
type SessionInfo struct {
	ID         string
	Name       string
	SocketPath string
}

// RendererSession manages a single renderer tmux session lifecycle.
type RendererSession struct {
	SessionID  string
	Name       string
	SocketPath string
	SocketDir  string
	ForwarderSocketPath string
	BinaryPath string
	Cols       int
	Rows       int
}

// NewRendererSession creates a new renderer session descriptor.
func NewRendererSession(sessionID, socketDir, forwarderSocketPath, binaryPath string, cols, rows int) *RendererSession {
	name := fmt.Sprintf("flashback-%s", sessionID)
	return &RendererSession{
		SessionID:           sessionID,
		Name:                name,
		SocketPath:          filepath.Join(socketDir, name),
		SocketDir:           socketDir,
		ForwarderSocketPath: forwarderSocketPath,
		BinaryPath:          binaryPath,
		Cols:                cols,
		Rows:                rows,
	}
}

// Start creates the renderer tmux session detached with kiosk mode settings.
func (s *RendererSession) Start() error {
	if err := os.MkdirAll(s.SocketDir, 0755); err != nil {
		return fmt.Errorf("create socket dir: %w", err)
	}

	// Write a minimal tmux config so global options are applied before the first
	// session is created.
	confPath := filepath.Join(s.SocketDir, "tmux.conf")
	conf := `# flashback-shell renderer tmux configuration
set -g status off
set -g mouse off
set -g default-terminal "xterm-256color"
set -g default-command ""
`
	if err := os.WriteFile(confPath, []byte(conf), 0644); err != nil {
		return fmt.Errorf("write tmux config: %w", err)
	}

	rendererCmd := fmt.Sprintf("%s __renderer %s", s.BinaryPath, s.ForwarderSocketPath)

	args := []string{
		"-S", s.SocketPath,
		"-f", confPath,
		"new-session", "-d",
		"-s", s.Name,
		"-n", "main",
		"-e", "TERM=xterm-256color",
		"-e", "COLORTERM=truecolor",
	}
	if s.Cols > 0 && s.Rows > 0 {
		args = append(args, "-x", fmt.Sprintf("%d", s.Cols), "-y", fmt.Sprintf("%d", s.Rows))
	}
	args = append(args, rendererCmd)

	cmd := exec.Command("tmux", args...)
	cmd.Env = s.env()
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("tmux new-session: %w\n%s", err, string(out))
	}

	// Disable all key bindings after the session is created (read-only mirror).
	c := exec.Command("tmux", "-S", s.SocketPath, "unbind-key", "-a")
	c.Env = s.env()
	c.CombinedOutput() // best-effort

	return nil
}

// Kill terminates the tmux session and removes the socket file.
func (s *RendererSession) Kill() error {
	cmd := exec.Command("tmux", "-S", s.SocketPath, "kill-session", "-t", s.Name)
	cmd.Env = s.env()
	cmd.CombinedOutput() // ignore errors; session may already be dead
	os.Remove(s.SocketPath)
	return nil
}

// IsRunning checks if the tmux session is still alive.
func (s *RendererSession) IsRunning() bool {
	cmd := exec.Command("tmux", "-S", s.SocketPath, "has-session", "-t", s.Name)
	cmd.Env = s.env()
	if cmd.Run() != nil {
		return false
	}
	_, err := os.Stat(s.SocketPath)
	return err == nil
}

// CapturePane captures the content of the renderer pane.
func (s *RendererSession) CapturePane(paneTarget string, withANSI bool) (string, error) {
	args := []string{"-S", s.SocketPath, "capture-pane", "-p", "-J", "-t", paneTarget}
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
func (s *RendererSession) ListPanes() ([]string, error) {
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

func (s *RendererSession) env() []string {
	var env []string
	for _, e := range os.Environ() {
		// Drop outer TERM/COLORTERM so we can force 256/truecolor support.
		if strings.HasPrefix(e, "TERM=") || strings.HasPrefix(e, "COLORTERM=") {
			continue
		}
		env = append(env, e)
	}
	env = append(env, "TERM=xterm-256color", "COLORTERM=truecolor")
	return env
}

// DiscoverSessions scans the socket directory for flashback-managed renderer
// sessions. Dead sessions (socket file exists but tmux has no matching session)
// are removed and omitted from the result.
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
		if entry.IsDir() || strings.HasSuffix(name, ".conf") || strings.HasSuffix(name, ".fwd.sock") {
			continue
		}
		if !strings.HasPrefix(name, "flashback-") {
			continue
		}
		id := strings.TrimPrefix(name, "flashback-")
		socketPath := filepath.Join(socketDir, name)

		// Skip stale sockets that no longer correspond to a live tmux session.
		sess := &RendererSession{Name: name, SocketPath: socketPath}
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

// HasTmux returns true if tmux is available in PATH.
func HasTmux() bool {
	_, err := exec.LookPath("tmux")
	return err == nil
}

// Path returns the absolute path to the tmux binary, or an error if it is not
// found in PATH.
func Path() (string, error) {
	return exec.LookPath("tmux")
}

// PrintInstallHint prints OS-specific tmux installation instructions.
func PrintInstallHint() {
	fmt.Fprintln(os.Stderr, "tmux not found in PATH.")
	switch os.Getenv("GOOS") {
	case "darwin":
		fmt.Fprintln(os.Stderr, "  Install: brew install tmux")
	case "linux":
		fmt.Fprintln(os.Stderr, "  Debian/Ubuntu: sudo apt-get install tmux")
		fmt.Fprintln(os.Stderr, "  RHEL/CentOS:   sudo yum install tmux")
		fmt.Fprintln(os.Stderr, "  Arch Linux:    sudo pacman -S tmux")
	default:
		fmt.Fprintln(os.Stderr, "  Please install tmux for your platform.")
	}
}
