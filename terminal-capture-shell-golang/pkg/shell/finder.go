package shell

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
)

// TmuxPath returns the absolute path to the tmux binary, or an error if it is
// not found in PATH.
func TmuxPath() (string, error) {
	return exec.LookPath("tmux")
}

// HasTmux returns true if tmux is available in PATH.
func HasTmux() bool {
	_, err := TmuxPath()
	return err == nil
}

// EnvPath returns the absolute path to the env(1) binary, or an error if it is
// not found in PATH.
func EnvPath() (string, error) {
	return exec.LookPath("env")
}

// HasEnv returns true if the env(1) binary is available in PATH.
// It is required for allow_nested_tmux because we use env -u to strip tmux
// variables from the inner shell process.
func HasEnv() bool {
	_, err := EnvPath()
	return err == nil
}

// FindFallbackShell returns the path to a usable shell binary.
// Tries: $SHELL → bash → sh
func FindFallbackShell() (string, error) {
	// Try configured shell
	if cfg := os.Getenv("SHELL"); cfg != "" {
		if _, err := exec.LookPath(cfg); err == nil {
			return cfg, nil
		}
	}
	// Try bash
	if p, err := exec.LookPath("bash"); err == nil {
		return p, nil
	}
	// Try sh
	if p, err := exec.LookPath("sh"); err == nil {
		return p, nil
	}
	return "", fmt.Errorf("no shell found in PATH (tried $SHELL, bash, sh)")
}

// PrintTmuxInstallHint prints OS-specific tmux installation instructions.
func PrintTmuxInstallHint() {
	fmt.Fprintln(os.Stderr, "tmux not found in PATH.")
	switch runtime.GOOS {
	case "darwin":
		fmt.Fprintln(os.Stderr, "  Install: brew install tmux")
	case "linux":
		fmt.Fprintln(os.Stderr, "  Debian/Ubuntu: sudo apt-get install tmux")
		fmt.Fprintln(os.Stderr, "  RHEL/CentOS:   sudo yum install tmux")
		fmt.Fprintln(os.Stderr, "  Arch Linux:    sudo pacman -S tmux")
	default:
		fmt.Fprintln(os.Stderr, "  Please install tmux for your platform.")
	}
	fmt.Fprintln(os.Stderr, "Falling back to plain shell...")
}
