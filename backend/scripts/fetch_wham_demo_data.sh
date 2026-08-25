#!/bin/bash
# macOS-friendly WHAM fetch_demo_data: uses curl when wget is missing.
# Run from vendor/WHAM (or pass path as $1).
set -euo pipefail

WHAM_DIR="${1:-}"
if [[ -z "$WHAM_DIR" ]]; then
  WHAM_DIR="$(cd "$(dirname "$0")/../.." && pwd)/vendor/WHAM"
fi
cd "$WHAM_DIR"

if ! command -v wget >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "Need wget or curl. On macOS: brew install wget" >&2
    exit 1
  fi
  # Drop-in wget shim for the few flags this script uses.
  wget() {
    local post_data="" out="" url="" continue=0 insecure=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --post-data) post_data="$2"; shift 2 ;;
        -O) out="$2"; shift 2 ;;
        --no-check-certificate) insecure=1; shift ;;
        --continue) continue=1; shift ;;
        --*) shift ;;
        *) url="$1"; shift ;;
      esac
    done
    local args=(-L)
    [[ $insecure -eq 1 ]] && args+=(-k)
    [[ $continue -eq 1 && -n "$out" && -f "$out" ]] && args+=(-C -)
    [[ -n "$out" ]] && args+=(-o "$out")
    [[ -n "$post_data" ]] && args+=(--data "$post_data")
    curl "${args[@]}" "$url"
  }
  export -f wget
fi

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown not found. Install with: pip install gdown" >&2
  exit 1
fi

exec bash ./fetch_demo_data.sh
