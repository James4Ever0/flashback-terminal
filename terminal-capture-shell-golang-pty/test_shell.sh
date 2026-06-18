#!/bin/bash

export FLASHBACK_SHELL_PTY_SERVER_URL=http://127.0.0.1:9090

./dist/flashback-shell-pty $@ 
