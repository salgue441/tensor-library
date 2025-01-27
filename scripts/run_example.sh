#!/usr/bin/env bash
set -euo pipefail

# Default values
BUILD_TYPE="Release"
COMPILER="gcc"

# Help message
show_help() {
  echo "Usage: $0 EXAMPLE_NAME [options]"
  echo "Options:"
  echo "  -h, --help              Show this help message"
  echo "  -t, --type TYPE         Build type (Debug|Release|RelWithDebInfo) [default: Release]"
  echo "  -c, --compiler CC       Compiler to use (gcc|clang) [default: gcc]"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
  -h | --help)
    show_help
    exit 0
    ;;
  -t | --type)
    BUILD_TYPE="$2"
    shift 2
    ;;
  -c | --compiler)
    COMPILER="$2"
    shift 2
    ;;
  *)
    if [ -z "${EXAMPLE_NAME+x}" ]; then
      EXAMPLE_NAME="$1"
      shift
    else
      echo "Unknown option: $1"
      show_help
      exit 1
    fi
    ;;
  esac
done

if [ -z "${EXAMPLE_NAME+x}" ]; then
  echo "Error: EXAMPLE_NAME is required"
  show_help
  exit 1
fi

BUILD_DIR="build/${COMPILER}-${BUILD_TYPE,,}"

if [ ! -d "$BUILD_DIR" ]; then
  echo "Build directory not found. Please run build.sh first."
  exit 1
fi

EXAMPLE_PATH="$BUILD_DIR/bin/$EXAMPLE_NAME"

if [ ! -f "$EXAMPLE_PATH" ]; then
  echo "Example '$EXAMPLE_NAME' not found in $EXAMPLE_PATH"
  exit 1
fi

echo "Running example '$EXAMPLE_NAME'..."
"$EXAMPLE_PATH"
