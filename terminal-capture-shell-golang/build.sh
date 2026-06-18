#!/usr/bin/env bash
# Build flashback-shell: cross-platform, CGO-free, static binary
#
# Usage:
#   ./build.sh              # Build for current platform
#   ./build.sh all          # Build for all platforms
#   GOOS=linux GOARCH=amd64 ./build.sh
#   ./build.sh windows      # Build for all Windows archs
#   ./build.sh darwin       # Build for all macOS archs
#   ./build.sh linux        # Build for all Linux archs

set -euo pipefail

MODULE="flashback-shell"
NEW_MODULE="flashback-shell-new"
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS="-s -w -X main.version=${VERSION}"

cd "$(dirname "$0")"

# Default: build for current platform
build_current() {
    local os=${GOOS:-$(go env GOOS)}
    local arch=${GOARCH:-$(go env GOARCH)}
    local suffix=""
    if [ "$os" = "windows" ]; then
        suffix=".exe"
    fi
    local out="dist/${MODULE}-${os}-${arch}${suffix}"
    local new_out="dist/${NEW_MODULE}-${os}-${arch}${suffix}"

    echo "Building ${MODULE} ${VERSION} for ${os}/${arch}..."
    mkdir -p dist
    CGO_ENABLED=0 go build \
        -ldflags "${LDFLAGS}" \
        -trimpath \
        -o "${out}" \
        .
    echo "  -> ${out}"
    ls -lh "${out}"

    echo "Building ${NEW_MODULE} ${VERSION} for ${os}/${arch}..."
    CGO_ENABLED=0 go build \
        -ldflags "${LDFLAGS}" \
        -trimpath \
        -o "${new_out}" \
        ./cmd/flashback-shell-new
    echo "  -> ${new_out}"
    ls -lh "${new_out}"
}

# Build for a specific OS/arch
build_one() {
    local os=$1
    local arch=$2
    local suffix=""
    if [ "$os" = "windows" ]; then
        suffix=".exe"
    fi
    local out="dist/${MODULE}-${os}-${arch}${suffix}"
    local new_out="dist/${NEW_MODULE}-${os}-${arch}${suffix}"

    echo "Building ${MODULE} ${VERSION} for ${os}/${arch}..."
    mkdir -p dist
    GOOS="$os" GOARCH="$arch" CGO_ENABLED=0 go build \
        -ldflags "${LDFLAGS}" \
        -trimpath \
        -o "${out}" \
        .
    echo "  -> ${out}"

    echo "Building ${NEW_MODULE} ${VERSION} for ${os}/${arch}..."
    GOOS="$os" GOARCH="$arch" CGO_ENABLED=0 go build \
        -ldflags "${LDFLAGS}" \
        -trimpath \
        -o "${new_out}" \
        ./cmd/flashback-shell-new
    echo "  -> ${new_out}"
}

# Build for all supported platforms
build_all() {
    echo "Building ${MODULE} ${VERSION} for all platforms..."
    mkdir -p dist

    # Linux
    build_one linux amd64
    build_one linux 386
    build_one linux arm64
    build_one linux arm

    # macOS
    build_one darwin amd64
    build_one darwin arm64

    # Windows
    build_one windows amd64
    build_one windows 386
    build_one windows arm64

    echo
    echo "All builds complete:"
    ls -lh dist/
}

# Build for a specific OS across archs
build_os() {
    local target=$1
    case $target in
        linux)
            build_one linux amd64
            build_one linux 386
            build_one linux arm64
            build_one linux arm
            ;;
        darwin|macos|osx)
            build_one darwin amd64
            build_one darwin arm64
            ;;
        windows|win)
            build_one windows amd64
            build_one windows 386
            build_one windows arm64
            ;;
        *)
            echo "Unknown OS: $target"
            echo "Supported: linux, darwin, windows"
            exit 1
            ;;
    esac
}

# Show help
show_help() {
    cat <<'EOF'
Build flashback-shell and flashback-shell-new: cross-platform, CGO-free, static binaries

Usage:
  ./build.sh              Build both binaries for current platform (default)
  ./build.sh all          Build both binaries for all supported platforms
  ./build.sh linux        Build for Linux (amd64, 386, arm64, arm)
  ./build.sh darwin       Build for macOS (amd64, arm64)
  ./build.sh windows      Build for Windows (amd64, 386, arm64)

Environment variables:
  GOOS      Target operating system (linux, darwin, windows, ...)
  GOARCH    Target architecture (amd64, 386, arm64, arm, ...)
  VERSION   Override version string (default: git tag or "dev")

The binaries are built with:
  - CGO_ENABLED=0       No C dependencies, fully static
  - -ldflags -s -w      Strip debug info and DWARF tables
  - -trimpath           Reproducible builds, no host paths in binary
EOF
}

main() {
    local cmd=${1:-current}
    case $cmd in
        -h|--help|help)
            show_help
            ;;
        all)
            build_all
            ;;
        linux|darwin|macos|osx|windows|win)
            build_os "$cmd"
            ;;
        current|"")
            build_current
            ;;
        *)
            echo "Unknown command: $cmd"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
