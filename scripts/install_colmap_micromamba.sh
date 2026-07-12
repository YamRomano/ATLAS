#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$ROOT/tools"
MAMBA_ROOT="$TOOLS/micromamba"
ENV_PREFIX="$TOOLS/colmap-env"

mkdir -p "$MAMBA_ROOT" "$TOOLS"

ARCH="$(uname -m)"
case "$ARCH" in
  arm64) PLATFORM="osx-arm64" ;;
  x86_64) PLATFORM="osx-64" ;;
  *) echo "Unsupported macOS architecture: $ARCH" >&2; exit 1 ;;
esac

MAMBA="$MAMBA_ROOT/bin/micromamba"
if [ ! -x "$MAMBA" ]; then
  echo "Downloading micromamba for $PLATFORM..."
  curl -L "https://micro.mamba.pm/api/micromamba/$PLATFORM/latest" | tar -xvj -C "$MAMBA_ROOT" bin/micromamba
fi

if [ ! -x "$ENV_PREFIX/bin/colmap" ]; then
  echo "Creating local COLMAP environment..."
  "$MAMBA" create -y -p "$ENV_PREFIX" -c conda-forge colmap
else
  echo "COLMAP environment already exists: $ENV_PREFIX"
fi

echo
echo "COLMAP binary:"
"$ENV_PREFIX/bin/colmap" -h | head -5 || true
echo
echo "Use this path in config.json:"
echo "$ENV_PREFIX/bin/colmap"
