// Package session implements the client/server protocol and socket discovery
// used to manage PTY-based terminal sessions.
package session

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
)

// RequestMethod is the set of commands a client can send to a session server.
type RequestMethod string

const (
	MethodAttach  RequestMethod = "attach"
	MethodCapture RequestMethod = "capture"
	MethodStatus  RequestMethod = "status"
	MethodKill    RequestMethod = "kill"
	MethodResize  RequestMethod = "resize"
)

// Request is the JSON envelope sent from client to server.
type Request struct {
	Method    RequestMethod `json:"method"`
	Cols      int           `json:"cols,omitempty"`
	Rows      int           `json:"rows,omitempty"`
	Scrollback bool         `json:"scrollback,omitempty"`
}

// Response is the JSON envelope sent from server to client.
type Response struct {
	OK      bool            `json:"ok"`
	Method  RequestMethod   `json:"method"`
	Error   string          `json:"error,omitempty"`
	Payload json.RawMessage `json:"payload,omitempty"`
}

// CapturePayload is returned by the "capture" method.
type CapturePayload struct {
	ANSI      string `json:"ansi"`
	Text      string `json:"text"`
	Cols      int    `json:"cols"`
	Rows      int    `json:"rows"`
	CursorRow int    `json:"cursor_row"`
	CursorCol int    `json:"cursor_col"`
}

// StatusPayload is returned by the "status" method.
type StatusPayload struct {
	SessionID string `json:"session_id"`
	ShellPID  int    `json:"shell_pid"`
	Cols      int    `json:"cols"`
	Rows      int    `json:"rows"`
	Attached  bool   `json:"attached"`
	Screen    string `json:"screen,omitempty"`
}

// WriteRequest writes a JSON request followed by a newline.
func WriteRequest(w io.Writer, req *Request) error {
	data, err := json.Marshal(req)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = w.Write(data)
	return err
}

// ReadResponse reads a single JSON response line.
func ReadResponse(r *bufio.Reader) (*Response, error) {
	line, err := r.ReadBytes('\n')
	if err != nil {
		return nil, err
	}
	var resp Response
	if err := json.Unmarshal(line, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ReadRequest reads a single JSON request line.
func ReadRequest(r *bufio.Reader) (*Request, error) {
	line, err := r.ReadBytes('\n')
	if err != nil {
		return nil, err
	}
	var req Request
	if err := json.Unmarshal(line, &req); err != nil {
		return nil, err
	}
	return &req, nil
}

// WriteResponse writes a JSON response followed by a newline.
func WriteResponse(w io.Writer, resp *Response) error {
	data, err := json.Marshal(resp)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = w.Write(data)
	return err
}

// ErrorResponse builds a failing response.
func ErrorResponse(method RequestMethod, err error) *Response {
	return &Response{OK: false, Method: method, Error: err.Error()}
}

// SuccessResponse builds a successful response with the given payload.
func SuccessResponse(method RequestMethod, payload interface{}) (*Response, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	return &Response{OK: true, Method: method, Payload: data}, nil
}

// SocketName returns the socket file name for a session ID.
func SocketName(sessionID string) string {
	return fmt.Sprintf("%s.sock", sessionID)
}
