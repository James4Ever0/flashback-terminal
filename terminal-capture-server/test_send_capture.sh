curl -s -X POST http://127.0.0.1:8080/api/captures \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "test-device",
    "timestamp": "2024-01-01T00:00:00Z",
    "captures": [{
      "session_id": "flashback-12345",
      "pane_id": "flashback-12345:%0",
      "target": "flashback-12345:%0",
      "ansi": "hello\nworld",
      "text": "hello\nworld",
      "hash": "a1b2c3d4e5f678901234567890123456",
      "cols": 0,
      "rows": 0,
      "timestamp": "2024-01-01T00:00:00Z"
    }]
  }'
