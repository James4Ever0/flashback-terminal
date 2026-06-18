package session

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Info describes a discovered session socket.
type Info struct {
	ID         string
	SocketPath string
	SessionID  string
}

// DiscoverSessions scans the socket directory and returns all session sockets.
func DiscoverSessions(socketDir string) ([]Info, error) {
	if err := os.MkdirAll(socketDir, 0755); err != nil {
		return nil, fmt.Errorf("create socket dir %s: %w", socketDir, err)
	}

	entries, err := os.ReadDir(socketDir)
	if err != nil {
		return nil, err
	}

	var sessions []Info
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasSuffix(name, ".sock") {
			continue
		}
		sessionID := strings.TrimSuffix(name, ".sock")
		sessions = append(sessions, Info{
			ID:         sessionID,
			SocketPath: filepath.Join(socketDir, name),
			SessionID:  sessionID,
		})
	}
	return sessions, nil
}

// SocketPath returns the full path for a session ID in the socket directory.
func SocketPath(socketDir, sessionID string) string {
	return filepath.Join(socketDir, SocketName(sessionID))
}

// RemoveStaleSocket removes a socket file if it exists.
func RemoveStaleSocket(path string) error {
	if _, err := os.Stat(path); err == nil {
		return os.Remove(path)
	}
	return nil
}
