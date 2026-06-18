package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"time"

	"flashback-shell/pkg/capture"
	"flashback-shell/pkg/config"
	"flashback-shell/pkg/log"
)

// ErrNoServerURL is returned when an upload is requested but no server URL is configured.
var ErrNoServerURL = errors.New("no server URL configured")

// Client uploads captures to the remote server.
type Client struct {
	cfg        *config.Config
	buffer     *capture.Buffer
	logger     *log.Logger
	httpClient *http.Client
}

// NewClient creates an upload client.
func NewClient(cfg *config.Config, buffer *capture.Buffer, logger *log.Logger) *Client {
	if logger == nil {
		logger = log.New(0, os.Stderr)
	}
	return &Client{
		cfg:        cfg,
		buffer:     buffer,
		logger:     logger,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

// Upload sends captures to the remote server.
func (c *Client) Upload(captures []capture.Capture) error {
	captureCount := len(captures)

	payload := uploadPayload{
		DeviceID:  deviceID(c.cfg),
		Timestamp: time.Now().UTC(),
		Captures:  captures,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		c.logger.Errorf("failed to marshal upload payload: %v", err)
		return err
	}
	bodyLen := len(body)

	if c.cfg.ServerURL == "" {
		c.logger.Infof("declining upload of %d capture(s) (%d bytes): no server URL configured (set FLASHBACK_SHELL_SERVER)", captureCount, bodyLen)
		return ErrNoServerURL
	}

	url := c.cfg.ServerURL + "/api/captures"
	c.logger.Infof("POST %s: sending %d capture(s), %d bytes", url, captureCount, bodyLen)

	// Exponential backoff retry
	var lastErr error
	const maxAttempts = 3
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			c.logger.Infof("retrying upload to %s (attempt %d/%d)", url, attempt+1, maxAttempts)
			time.Sleep(time.Duration(attempt) * 2 * time.Second)
		}
		c.logger.Debugf("upload request to %s: attempt %d/%d, %d bytes", url, attempt+1, maxAttempts, bodyLen)

		req, err := http.NewRequest("POST", url, bytes.NewReader(body))
		if err != nil {
			c.logger.Errorf("failed to build upload request to %s: %v", url, err)
			return err
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = err
			c.logger.Warnf("upload attempt %d/%d to %s failed: %v", attempt+1, maxAttempts, url, err)
			continue
		}
		resp.Body.Close()

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			c.logger.Infof("upload to %s succeeded: %s (%d bytes, %d capture(s))", url, resp.Status, bodyLen, captureCount)
			return nil
		}

		lastErr = fmt.Errorf("server returned %d", resp.StatusCode)
		c.logger.Warnf("upload attempt %d/%d to %s returned %s", attempt+1, maxAttempts, url, resp.Status)
	}

	c.logger.Errorf("upload to %s failed after %d attempts: %v", url, maxAttempts, lastErr)

	// Buffer for retry
	if c.buffer != nil {
		if err := c.buffer.Add(captures); err != nil {
			c.logger.Errorf("failed to buffer captures for retry: %v", err)
		} else {
			c.logger.Infof("buffered %d capture(s) for retry", captureCount)
		}
	}
	return fmt.Errorf("upload failed after retries: %w", lastErr)
}

// FlushRetries attempts to upload any buffered captures.
func (c *Client) FlushRetries() error {
	if c.buffer == nil {
		return nil
	}
	batches, err := c.buffer.PopAll()
	if err != nil {
		return err
	}
	if len(batches) == 0 {
		c.logger.Debugf("no buffered captures to flush")
		return nil
	}
	c.logger.Infof("flushing %d buffered capture batch(es)", len(batches))
	for _, batch := range batches {
		if err := c.Upload(batch); err != nil {
			// Re-buffer on failure
			_ = c.buffer.Add(batch)
			return err
		}
	}
	return nil
}

type uploadPayload struct {
	DeviceID  string            `json:"device_id"`
	Timestamp time.Time         `json:"timestamp"`
	Captures  []capture.Capture `json:"captures"`
}

func deviceID(cfg *config.Config) string {
	if cfg.DeviceID != "" {
		return cfg.DeviceID
	}
	host, _ := os.Hostname()
	return host
}
