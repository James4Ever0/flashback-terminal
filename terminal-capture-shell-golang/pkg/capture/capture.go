package capture

import (
	"crypto/md5"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"flashback-shell/pkg/shell"
)

// Capture holds the captured content of a single pane.
type Capture struct {
	SessionID string            `json:"session_id"`
	PaneID    string            `json:"pane_id"`
	Target    string            `json:"target"`
	ANSI      string            `json:"ansi"`
	Text      string            `json:"text"`
	Hash      string            `json:"hash"`
	Cols      int               `json:"cols"`
	Rows      int               `json:"rows"`
	Timestamp time.Time         `json:"timestamp"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

// Engine captures terminal content from all managed tmux sessions.
type Engine struct {
	stateDir          string
	CaptureScrollback bool
	AllowNestedTmux   bool
	DiffOnly          bool
	DiffMode          string // "suffix" (default) or "index"
	TextOnly          bool   // when true, send plain text only and omit ANSI
}

// NewEngine creates a capture engine.
func NewEngine(stateDir string) *Engine {
	return &Engine{stateDir: stateDir}
}

// CaptureAll discovers all sessions and captures changed panes.
func (e *Engine) CaptureAll(socketDir string) ([]Capture, error) {
	sessions, err := shell.DiscoverSessions(socketDir)
	if err != nil {
		return nil, err
	}

	var results []Capture
	for _, si := range sessions {
		sess := &shell.TmuxSession{
			Name:              si.Name,
			SocketPath:        si.SocketPath,
			CaptureScrollback: e.CaptureScrollback,
			AllowNestedTmux:   e.AllowNestedTmux,
		}
		panes, err := sess.ListPanes()
		if err != nil {
			continue // skip dead sessions
		}

		for _, target := range panes {
			c, err := e.capturePane(sess, si.ID, target)
			if err != nil {
				continue
			}
			if c != nil {
				results = append(results, *c)
			}
		}
	}
	return results, nil
}

// CaptureSession captures all panes of a specific tmux session.
func (e *Engine) CaptureSession(sess *shell.TmuxSession) ([]Capture, error) {
	panes, err := sess.ListPanes()
	if err != nil {
		return nil, err
	}

	var results []Capture
	for _, target := range panes {
		c, err := e.capturePane(sess, sess.SessionID, target)
		if err != nil {
			continue
		}
		if c != nil {
			results = append(results, *c)
		}
	}
	return results, nil
}

// capturePane captures a single pane, applying deduplication and optional
// diff-only / text-only logic. It returns nil when the pane should be skipped.
func (e *Engine) capturePane(sess *shell.TmuxSession, sessionID, target string) (*Capture, error) {
	var ansi, text string
	var err error

	if e.TextOnly {
		// Text-only mode: skip ANSI capture entirely. The deduplication hash
		// is computed over the plain text so that unchanged panes are still
		// filtered correctly.
		text, err = sess.CapturePane(target, false)
		if err != nil {
			return nil, err
		}
	} else {
		ansi, err = sess.CapturePane(target, true)
		if err != nil {
			return nil, err
		}
		text, _ = sess.CapturePane(target, false)
	}

	hashInput := ansi
	if e.TextOnly {
		hashInput = text
	}
	hash := fmt.Sprintf("%x", md5.Sum([]byte(hashInput)))
	if e.isDuplicate(sessionID, target, hash) {
		return nil, nil
	}

	captureANSI := ansi
	captureText := text

	if e.DiffOnly {
		prevText, err := e.loadPrev(sessionID, target)
		if err == nil && len(prevText) > 0 {
			prevLines := strings.Split(prevText, "\n")
			currLines := strings.Split(text, "\n")

			var diff []string
			var indices []int
			switch strings.ToLower(e.DiffMode) {
			case "index":
				// Index-wise diff: align the previous buffer to the current
				// length by trimming or padding at the top, then keep every
				// line that changed at its vertical position.
				diff, indices = diffLinesIndex(prevLines, currLines)
			default:
				// Suffix diff (default): assume new lines are appended at the
				// bottom and old lines scroll off the top. Find the longest
				// overlap between the end of the previous capture and the
				// beginning of the current capture.
				diff, indices = diffLines(prevLines, currLines)
			}

			if len(diff) == 0 {
				// Content changed, but no new/different lines appeared (e.g.
				// in-place updates). Update the baseline and skip this capture.
				_ = e.savePrev(sessionID, target, text)
				return nil, nil
			}

			captureText = strings.Join(diff, "\n")
			ansiLines := strings.Split(ansi, "\n")
			if len(ansiLines) == len(currLines) {
				selected := make([]string, 0, len(indices))
				for _, idx := range indices {
					selected = append(selected, ansiLines[idx])
				}
				captureANSI = strings.Join(selected, "\n")
			}
		}
		// Always store the full current text as the next baseline.
		_ = e.savePrev(sessionID, target, text)
	}

	var metadata map[string]string
	if e.TextOnly {
		metadata = map[string]string{"ansi": "false"}
	}

	return &Capture{
		SessionID: sessionID,
		PaneID:    target,
		Target:    target,
		ANSI:      captureANSI,
		Text:      captureText,
		Hash:      hash,
		Timestamp: time.Now().UTC(),
		Metadata:  metadata,
	}, nil
}

// diffLines returns the suffix of curr that did not exist as a contiguous
// block at the end of prev. This models a ring-like scrollback where old lines
// scroll off the top and new lines are appended at the bottom.
// If there is no overlap, the full curr slice is returned.
// It also returns the indices of the returned lines within curr so the ANSI
// variant can be sliced in the same way.
func diffLines(prev, curr []string) ([]string, []int) {
	max := len(prev)
	if len(curr) < max {
		max = len(curr)
	}
	for i := max; i > 0; i-- {
		if slices.Equal(prev[len(prev)-i:], curr[:i]) {
			return curr[i:], makeRange(i, len(curr))
		}
	}
	return curr, makeRange(0, len(curr))
}

// diffLinesIndex aligns prev to the length of curr by trimming the oldest
// lines (when prev is longer) or padding with empty strings at the top (when
// prev is shorter), then returns every line in curr that differs from the
// aligned previous line at the same index. This is useful when the terminal
// updates in place rather than only appending at the bottom.
func diffLinesIndex(prev, curr []string) ([]string, []int) {
	aligned := make([]string, len(curr))
	switch {
	case len(prev) > len(curr):
		// Keep only the most recent lines from the previous capture.
		copy(aligned, prev[len(prev)-len(curr):])
	case len(prev) < len(curr):
		// The previous capture was shorter; pad the top with empty lines.
		copy(aligned[len(curr)-len(prev):], prev)
	default:
		copy(aligned, prev)
	}

	var diff []string
	var indices []int
	for i, line := range curr {
		if line != aligned[i] {
			diff = append(diff, line)
			indices = append(indices, i)
		}
	}
	return diff, indices
}

// makeRange returns a slice of integers in [start, end). If start == end, the
// returned slice is empty.
func makeRange(start, end int) []int {
	if start >= end {
		return nil
	}
	r := make([]int, end-start)
	for i := range r {
		r[i] = start + i
	}
	return r
}

func (e *Engine) prevPath(sessionID, paneID string) string {
	key := hashKey(sessionID, paneID)
	return filepath.Join(e.stateDir, key+".prev")
}

func (e *Engine) loadPrev(sessionID, paneID string) (string, error) {
	data, err := os.ReadFile(e.prevPath(sessionID, paneID))
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (e *Engine) savePrev(sessionID, paneID, text string) error {
	if err := os.MkdirAll(e.stateDir, 0755); err != nil {
		return err
	}
	return os.WriteFile(e.prevPath(sessionID, paneID), []byte(text), 0644)
}

// SaveHashes writes the last-seen hashes for captured panes to disk.
func (e *Engine) SaveHashes(captures []Capture) error {
	if err := os.MkdirAll(e.stateDir, 0755); err != nil {
		return err
	}
	for _, c := range captures {
		key := hashKey(c.SessionID, c.PaneID)
		path := filepath.Join(e.stateDir, key+".hash")
		if err := os.WriteFile(path, []byte(c.Hash), 0644); err != nil {
			return err
		}
	}
	return nil
}

func (e *Engine) isDuplicate(sessionID, paneID, hash string) bool {
	key := hashKey(sessionID, paneID)
	path := filepath.Join(e.stateDir, key+".hash")
	data, err := os.ReadFile(path)
	if err != nil {
		return false // no previous hash = not a duplicate
	}
	return string(data) == hash
}

func hashKey(sessionID, paneID string) string {
	// Sanitize for filesystem
	return fmt.Sprintf("%s_%s", sessionID, paneID)
}
