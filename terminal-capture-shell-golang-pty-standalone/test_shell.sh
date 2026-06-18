#!/bin/bash

export FLASHBACK_SHELL_PTY_SERVER_URL=http://127.0.0.1:9090
export FLASHBACK_SHELL_PTY_FIRST_CAPTURE_DELAY=5

./dist/flashback-shell-pty $@ 
