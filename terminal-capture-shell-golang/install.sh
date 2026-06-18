#!/usr/bin/env bash
# install.sh — Install flashback-shell binaries and configure VS Code.
#
# Usage:
#   ./install.sh                  # Install to ~/.local/bin
#   INSTALL_DIR=/usr/local/bin ./install.sh   # Install to a custom location
#
# Supports Linux and macOS only. Windows must be configured manually.

set -euo pipefail

MODULE="flashback-shell"
NEW_MODULE="flashback-shell-new"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

OS=$(uname -s)
ARCH=$(uname -m)

case "$OS" in
    Linux)
        PLATFORM="linux"
        ;;
    Darwin)
        PLATFORM="darwin"
        ;;
    *)
        echo "Unsupported operating system: $OS"
        echo "This installer supports Linux and macOS only."
        echo "For Windows, place the binaries on PATH and update VS Code settings manually."
        exit 1
        ;;
esac

case "$ARCH" in
    x86_64)
        GOARCH="amd64"
        ;;
    aarch64|arm64)
        GOARCH="arm64"
        ;;
    i386|i686)
        GOARCH="386"
        ;;
    armv7l)
        GOARCH="arm"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

echo "Building flashback-shell for ${PLATFORM}/${GOARCH}..."
./build.sh current

echo "Installing binaries to ${INSTALL_DIR}..."
mkdir -p "${INSTALL_DIR}"
cp "dist/${MODULE}-${PLATFORM}-${GOARCH}" "${INSTALL_DIR}/${MODULE}"
cp "dist/${NEW_MODULE}-${PLATFORM}-${GOARCH}" "${INSTALL_DIR}/${NEW_MODULE}"
chmod +x "${INSTALL_DIR}/${MODULE}" "${INSTALL_DIR}/${NEW_MODULE}"

# Warn if the install directory is not on PATH, because VS Code expects to find
# "flashback-shell-new" by name.
case ":${PATH}:" in
    *:"${INSTALL_DIR}":*)
        ;;
    *)
        echo
        echo "Warning: ${INSTALL_DIR} is not on your PATH."
        echo "Add the following line to your shell profile so VS Code can find the binary:"
        echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
        echo
        ;;
esac

case "$PLATFORM" in
    linux)
        SETTINGS="${HOME}/.config/Code/User/settings.json"
        PROFILE_KEY="terminal.integrated.profiles.linux"
        DEFAULT_KEY="terminal.integrated.defaultProfile.linux"
        ;;
    darwin)
        SETTINGS="${HOME}/Library/Application Support/Code/User/settings.json"
        PROFILE_KEY="terminal.integrated.profiles.osx"
        DEFAULT_KEY="terminal.integrated.defaultProfile.osx"
        ;;
esac

if ! command -v python3 >/dev/null 2>&1; then
    echo
    echo "python3 not found; the binaries are installed, but you must update"
    echo "VS Code settings manually at:"
    echo "  ${SETTINGS}"
    echo
    echo "Add the following configuration (merging with existing settings):"
    echo
    cat <<EOF
{
    "${PROFILE_KEY}": {
        "flashback-shell": {
            "path": "flashback-shell-new",
            "args": [],
            "env": {}
        }
    },
    "${DEFAULT_KEY}": "flashback-shell"
}
EOF
    echo
    exit 0
fi

echo "Updating VS Code settings: ${SETTINGS}"
mkdir -p "$(dirname "${SETTINGS}")"
python3 - "${SETTINGS}" "${PROFILE_KEY}" "${DEFAULT_KEY}" <<'PY'
import json, os, sys

path = sys.argv[1]
profile_key = sys.argv[2]
default_key = sys.argv[3]

data = {}
if os.path.exists(path):
    with open(path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing {path}: {e}", file=sys.stderr)
            sys.exit(1)

data[profile_key] = data.get(profile_key, {})
data[profile_key]["flashback-shell"] = {
    "path": "flashback-shell",
    "args": ["new"],
    "env": {}
}
data[default_key] = "flashback-shell"

with open(path, "w") as f:
    json.dump(data, f, indent=4)

print(f"Updated {path}")
PY

echo
echo "Installation complete."
echo "  Binaries: ${INSTALL_DIR}/{${MODULE},${NEW_MODULE}}"
echo "  VS Code settings: ${SETTINGS}"
echo "Restart VS Code terminal panels to use flashback-shell."
