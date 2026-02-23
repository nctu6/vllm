#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: install_deps.sh [--no-mpi] [--with-mpi] [--no-ucx] [--with-ucx]

Installs TurboMind build/runtime dependencies for Debian/Ubuntu.
This script intentionally does not install TensorRT packages.

Flags:
  --with-mpi   Enable MPI dependencies (default)
  --no-mpi     Disable MPI dependencies
  --with-ucx   Enable UCX dependencies (default)
  --no-ucx     Disable UCX dependencies

Environment:
  VLLM_TLLM_ENABLE_UCX           ON/OFF override for TurboMind UCX feature.
  VLLM_TLLM_ENABLE_MULTI_DEVICE  ON/OFF override for TurboMind multi-device feature.
  VLLM_TLLM_ENABLE_NVSHMEM       ON/OFF override for NVSHMEM feature (default OFF).
USAGE
}

WITH_MPI=1
WITH_UCX=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-mpi)
      WITH_MPI=1
      ;;
    --no-mpi)
      WITH_MPI=0
      ;;
    --with-ucx)
      WITH_UCX=1
      ;;
    --no-ucx)
      WITH_UCX=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

case "${VLLM_TLLM_ENABLE_UCX:-ON}" in
  1|[Oo][Nn]|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]) VLLM_TLLM_ENABLE_UCX=ON ;;
  *) VLLM_TLLM_ENABLE_UCX=OFF ;;
esac

case "${VLLM_TLLM_ENABLE_MULTI_DEVICE:-ON}" in
  1|[Oo][Nn]|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]) VLLM_TLLM_ENABLE_MULTI_DEVICE=ON ;;
  *) VLLM_TLLM_ENABLE_MULTI_DEVICE=OFF ;;
esac

case "${VLLM_TLLM_ENABLE_NVSHMEM:-OFF}" in
  1|[Oo][Nn]|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]) VLLM_TLLM_ENABLE_NVSHMEM=ON ;;
  *) VLLM_TLLM_ENABLE_NVSHMEM=OFF ;;
esac

if [[ ${WITH_UCX} -eq 1 ]]; then
  VLLM_TLLM_ENABLE_UCX=ON
else
  VLLM_TLLM_ENABLE_UCX=OFF
fi

if [[ ${WITH_MPI} -eq 1 ]]; then
  VLLM_TLLM_ENABLE_MULTI_DEVICE=ON
else
  VLLM_TLLM_ENABLE_MULTI_DEVICE=OFF
fi

export VLLM_TLLM_ENABLE_UCX
export VLLM_TLLM_ENABLE_MULTI_DEVICE
export VLLM_TLLM_ENABLE_NVSHMEM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get not found. This script is for Debian/Ubuntu."
  exit 1
fi

apt-get update

# Ensure NCCL provides window APIs (ncclWindow_t, ncclCommWindowRegister).
apt-mark unhold libnccl2 libnccl-dev || true
apt-get install -y \
  libnccl2=2.27.7-1+cuda12.9 \
  libnccl-dev=2.27.7-1+cuda12.9

pkgs=(
  ca-certificates
  curl
  git
  cmake
  ninja-build
  libnuma-dev
  libzmq3-dev
  pkg-config
  autoconf
  automake
  libtool
  make
  g++
  ripgrep
)

if [[ ${WITH_MPI} -eq 1 ]]; then
  pkgs+=(
    openmpi-bin
    libopenmpi-dev
  )
fi

if [[ ${WITH_UCX} -eq 1 ]]; then
  pkgs+=(
    libucx-dev
    ucx-utils
  )
fi

apt-get install -y "${pkgs[@]}"

ensure_nvtx_headers() {
  if [[ -f "/usr/local/cuda/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/cuda-12.8/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/cuda-12.8/targets/x86_64-linux/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/include/nvtx3/nvtx3.hpp" ]]; then
    return 0
  fi

  echo "NVTX headers not found. Installing NVTX package..."
  apt-get install -y cuda-nvtx-12-8 || true
  apt-get install -y cuda-nvtx-dev-12-8 || true
  apt-get install -y libnvtx3-dev || true
  apt-get install -y libnvtx-dev || true

  if [[ -f "/usr/local/cuda/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/cuda-12.8/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/cuda-12.8/targets/x86_64-linux/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/include/nvtx3/nvtx3.hpp" ]]; then
    return 0
  fi

  echo "NVTX headers still missing after apt. Installing NVTX headers from GitHub..."
  NVTX_REPO="https://github.com/NVIDIA/NVTX.git"
  NVTX_TAG="v3.1.0"
  TMP_NVTX_DIR="/tmp/nvtx-src"
  rm -rf "${TMP_NVTX_DIR}"
  git clone --depth 1 -b "${NVTX_TAG}" "${NVTX_REPO}" "${TMP_NVTX_DIR}"
  if [[ -d "${TMP_NVTX_DIR}/c/include/nvtx3" ]]; then
    mkdir -p "/usr/local/include/nvtx3"
    cp -a "${TMP_NVTX_DIR}/c/include/nvtx3/." "/usr/local/include/nvtx3/"
  fi
  rm -rf "${TMP_NVTX_DIR}"

  if [[ -d "/usr/local/cuda/include" ]]; then
    mkdir -p "/usr/local/cuda/include/nvtx3"
    cp -f "/usr/local/include/nvtx3/nvtx3.hpp" "/usr/local/cuda/include/nvtx3/nvtx3.hpp"
  fi
  if [[ -d "/usr/local/cuda-12.8/include" ]]; then
    mkdir -p "/usr/local/cuda-12.8/include/nvtx3"
    cp -f "/usr/local/include/nvtx3/nvtx3.hpp" "/usr/local/cuda-12.8/include/nvtx3/nvtx3.hpp"
  fi

  if [[ -f "/usr/local/cuda/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/cuda-12.8/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/cuda-12.8/targets/x86_64-linux/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/include/nvtx3/nvtx3.hpp" ]] \
    || [[ -f "/usr/local/include/nvtx3/nvtx3.hpp" ]]; then
    return 0
  fi

  echo "Error: NVTX headers still missing. Ensure nvtx3.hpp is available in your CUDA install."
  exit 1
}

ensure_nvtx_headers

version_ge() {
  local ver_a="$1"
  local ver_b="$2"
  if [[ -z "${ver_a}" ]]; then
    return 1
  fi
  if [[ "${ver_a}" == "${ver_b}" ]]; then
    return 0
  fi
  local sorted
  sorted="$(printf '%s\n' "${ver_a}" "${ver_b}" | sort -V | head -n1)"
  [[ "${sorted}" == "${ver_b}" ]]
}

if [[ ${WITH_UCX} -eq 1 ]]; then
  UCX_REQUIRED_VERSION="1.20.0"
  UCX_VERSION_INSTALLED=""
  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists ucx; then
    UCX_VERSION_INSTALLED="$(pkg-config --modversion ucx || true)"
  fi

  if ! version_ge "${UCX_VERSION_INSTALLED}" "${UCX_REQUIRED_VERSION}"; then
    echo "Building UCX from source (required >= ${UCX_REQUIRED_VERSION}, found '${UCX_VERSION_INSTALLED:-none}')"
    UCX_REPO="https://github.com/openucx/ucx.git"
    UCX_BRANCH="v1.20.x"
    UCX_COMMIT="f656dbdf93e72e60b5d6ca78b9e3d9e744e789bd"
    UCX_INSTALL_PATH="/usr/local/ucx"
    CUDA_PATH="/usr/local/cuda"
    TMP_UCX_DIR="/tmp/ucx-src"
    rm -rf "${TMP_UCX_DIR}"
    git clone -b "${UCX_BRANCH}" "${UCX_REPO}" "${TMP_UCX_DIR}"
    (cd "${TMP_UCX_DIR}" && git checkout "${UCX_COMMIT}")
    (cd "${TMP_UCX_DIR}" && ./autogen.sh)
    (cd "${TMP_UCX_DIR}" && ./contrib/configure-release \
      --prefix="${UCX_INSTALL_PATH}" \
      --enable-shared \
      --disable-static \
      --disable-doxygen-doc \
      --enable-optimizations \
      --enable-cma \
      --enable-devel-headers \
      --with-cuda="${CUDA_PATH}" \
      --with-verbs \
      --with-dm \
      --enable-mt)
    (cd "${TMP_UCX_DIR}" && make -j"$(nproc)" install)
    rm -rf "${TMP_UCX_DIR}"
  fi

  if [[ -d "/usr/local/ucx" ]]; then
    export UCX_ROOT="/usr/local/ucx"
    for libdir in "${UCX_ROOT}/lib" "${UCX_ROOT}/lib64"; do
      if [[ -d "${libdir}" ]]; then
        export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH:-}"
        export LIBRARY_PATH="${libdir}:${LIBRARY_PATH:-}"
        export PKG_CONFIG_PATH="${libdir}/pkgconfig:${PKG_CONFIG_PATH:-}"
      fi
    done
  fi
fi

python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128 --no-input --exists-action i
python -m pip install -U pip packaging setuptools wheel setuptools-scm --no-input --exists-action i
if [[ ${WITH_MPI} -eq 1 ]]; then
  python -m pip install -U mpi4py --no-input --exists-action i
fi

cat <<'EOF_NOTES'

Notes:
- If MPI is disabled, TurboMind multi-device support is disabled:
  export VLLM_TLLM_ENABLE_MULTI_DEVICE=OFF
- If UCX is disabled, TurboMind UCX support is disabled:
  export VLLM_TLLM_ENABLE_UCX=OFF
EOF_NOTES
