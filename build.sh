#!/usr/bin/env bash
# build.sh — One-command build for the Ternary GGUF converter (t81z)
# Works on Linux, macOS, WSL, and GitHub Actions
set -euo pipefail

GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}=== Ternary LLM Converter Build Script ===${NC}"

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"
echo "Detected: $OS $ARCH"

# Find C++ compiler
if command -v clang++ >/dev/null; then
    CXX="clang++"
elif command -v g++ >/dev/null; then
    CXX="g++"
else
    echo -e "${RED}Error: No suitable C++ compiler found (need g++ or clang++)${NC}"
    exit 1
fi

echo "Using compiler: $CXX"

# Build flags: max performance + native
BASE_FLAGS="-std=c++20 -O3 -march=native -flto -DNDEBUG"
WARN_FLAGS="-Wall -Wextra -Wpedantic"

# macOS needs special treatment
if [[ "$OS" == "Darwin" ]]; then
    BASE_FLAGS="$BASE_FLAGS -stdlib=libc++"
    LDFLAGS="-lc++"
else
    LDFLAGS="-static-libgcc -static-libstdc++"
fi

# Final command
CMD="$CXX src/t81z.cpp $BASE_FLAGS $WARN_FLAGS -lz $LDFLAGS -o t81z"

echo -e "${CYAN}Building with maximum optimization...${NC}"
echo "$CMD"
echo

# Build it
$CMD

# Success message
echo
echo -e "${GREEN}Build successful!${NC}"
echo -e "${GREEN}Binary: ./t81z${NC}"
echo
echo "Example usage:"
echo "  ./t81z gemma-2b-it-safetensors/ --to-gguf gemma-2b-t3.gguf"
echo
echo "Run './t81z --help' for full options"
echo

# Show binary info
if command -v strip >/dev/null; then
    strip t81z 2>/dev/null || true
fi

echo -e "${GREEN}Ready for ternary domination.${NC}"
