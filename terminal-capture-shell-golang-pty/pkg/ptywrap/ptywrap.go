// Package ptywrap wraps a command in a PTY and resizes it.
package ptywrap

import (
	"os"
	"os/exec"

	"github.com/creack/pty"
)

// Start runs cmd attached to a PTY and returns the PTY master.
func Start(cmd *exec.Cmd) (*os.File, error) {
	ptmx, err := pty.Start(cmd)
	if err != nil {
		return nil, err
	}
	return ptmx, nil
}

// Resize sets the PTY size using TIOCSWINSZ.
func Resize(ptmx *os.File, rows, cols uint16) error {
	return pty.Setsize(ptmx, &pty.Winsize{Rows: rows, Cols: cols})
}

// Size returns the current PTY size.
func Size(ptmx *os.File) (*pty.Winsize, error) {
	ws, err := pty.GetsizeFull(ptmx)
	if err != nil {
		return nil, err
	}
	return ws, nil
}
