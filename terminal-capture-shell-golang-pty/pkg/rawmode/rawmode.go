// Package rawmode switches the controlling terminal into raw mode and restores it.
package rawmode

import (
	"golang.org/x/term"
)

// State holds the terminal state returned by MakeRaw.
type State = term.State

// MakeRaw puts the given fd into raw mode and returns a restore function.
func MakeRaw(fd int) (restore func() error, state *term.State, err error) {
	old, err := term.MakeRaw(fd)
	if err != nil {
		return nil, nil, err
	}
	restore = func() error {
		return term.Restore(fd, old)
	}
	return restore, old, nil
}

// IsTerminal reports whether the given file descriptor is a terminal.
func IsTerminal(fd int) bool {
	return term.IsTerminal(fd)
}

// Size returns the terminal size of the given fd as (cols, rows).
func Size(fd int) (cols, rows int, err error) {
	return term.GetSize(fd)
}
