package session

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"time"
)

// Client communicates with a session server over its Unix socket.
type Client struct {
	socketPath string
	dialTimeout time.Duration
}

// NewClient creates a client for the given socket path.
func NewClient(socketPath string) *Client {
	return &Client{
		socketPath:  socketPath,
		dialTimeout: 2 * time.Second,
	}
}

// connect dials the session server and returns a connection and buffered reader.
func (c *Client) connect() (net.Conn, *bufio.Reader, error) {
	conn, err := net.DialTimeout("unix", c.socketPath, c.dialTimeout)
	if err != nil {
		return nil, nil, err
	}
	return conn, bufio.NewReader(conn), nil
}

// Attach connects to a session and bridges stdin/stdout over the connection.
// The caller is responsible for the returned rawConn; the response is returned
// first so the caller can confirm the attach succeeded before entering raw mode.
func (c *Client) Attach(cols, rows int) (net.Conn, *StatusPayload, error) {
	conn, reader, err := c.connect()
	if err != nil {
		return nil, nil, err
	}
	req := &Request{Method: MethodAttach, Cols: cols, Rows: rows}
	if err := WriteRequest(conn, req); err != nil {
		conn.Close()
		return nil, nil, err
	}
	resp, err := ReadResponse(reader)
	if err != nil {
		conn.Close()
		return nil, nil, err
	}
	if !resp.OK {
		conn.Close()
		return nil, nil, fmt.Errorf("attach failed: %s", resp.Error)
	}
	var payload StatusPayload
	if err := json.Unmarshal(resp.Payload, &payload); err != nil {
		conn.Close()
		return nil, nil, err
	}
	return conn, &payload, nil
}

// Capture requests a screen capture from the session.
func (c *Client) Capture(scrollback bool) (*CapturePayload, error) {
	conn, reader, err := c.connect()
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	req := &Request{Method: MethodCapture, Scrollback: scrollback}
	if err := WriteRequest(conn, req); err != nil {
		return nil, err
	}
	resp, err := ReadResponse(reader)
	if err != nil {
		return nil, err
	}
	if !resp.OK {
		return nil, fmt.Errorf("capture failed: %s", resp.Error)
	}
	var payload CapturePayload
	if err := json.Unmarshal(resp.Payload, &payload); err != nil {
		return nil, err
	}
	return &payload, nil
}

// Status requests session metadata from the server.
func (c *Client) Status() (*StatusPayload, error) {
	conn, reader, err := c.connect()
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	req := &Request{Method: MethodStatus}
	if err := WriteRequest(conn, req); err != nil {
		return nil, err
	}
	resp, err := ReadResponse(reader)
	if err != nil {
		return nil, err
	}
	if !resp.OK {
		return nil, fmt.Errorf("status failed: %s", resp.Error)
	}
	var payload StatusPayload
	if err := json.Unmarshal(resp.Payload, &payload); err != nil {
		return nil, err
	}
	return &payload, nil
}

// Kill requests the server to terminate the session.
func (c *Client) Kill() error {
	conn, reader, err := c.connect()
	if err != nil {
		return err
	}
	defer conn.Close()

	req := &Request{Method: MethodKill}
	if err := WriteRequest(conn, req); err != nil {
		return err
	}
	resp, err := ReadResponse(reader)
	if err != nil {
		return err
	}
	if !resp.OK {
		return fmt.Errorf("kill failed: %s", resp.Error)
	}
	return nil
}

// Resize requests the server to resize the PTY and VT emulator.
func (c *Client) Resize(cols, rows int) (*StatusPayload, error) {
	conn, reader, err := c.connect()
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	req := &Request{Method: MethodResize, Cols: cols, Rows: rows}
	if err := WriteRequest(conn, req); err != nil {
		return nil, err
	}
	resp, err := ReadResponse(reader)
	if err != nil {
		return nil, err
	}
	if !resp.OK {
		return nil, fmt.Errorf("resize failed: %s", resp.Error)
	}
	var payload StatusPayload
	if err := json.Unmarshal(resp.Payload, &payload); err != nil {
		return nil, err
	}
	return &payload, nil
}
