#!/usr/bin/env bash
set -euo pipefail

# The container runs as your host uid:gid so that files written into the
# bind-mounted repo stay owned by you. /root is not writable to that uid, so HOME
# is pointed at a directory inside the mount instead.
export HOME="${HOME:-/workspace/ImageWAM/.docker-home}"
mkdir -p "${HOME}"

# torch.compile / inductor and triton both scribble into cache dirs at import
# time; keep them on the mount so they survive `--rm` and never land somewhere
# read-only.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${HOME}/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${HOME}/.cache/triton}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

# Mesa derives its shader cache from the passwd home, not $HOME, and warns
# loudly on every render when it cannot write there. Point it at the mount.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
mkdir -p "${XDG_CACHE_HOME}"

# `import libero` calls input() to ask "specify a custom path for the dataset
# folder?" whenever ~/.libero/config.yaml is absent -- an EOFError in any
# non-interactive container. Seed the config with container paths so the import
# is silent and deterministic. (The host's copy of this file points at
# /data3/..., which does not resolve in here.)
LIBERO_ROOT="${LIBERO_ROOT:-/workspace/ImageWAM/third_party/LIBERO/libero/libero}"
LIBERO_CONFIG="${HOME}/.libero/config.yaml"
if [ ! -f "${LIBERO_CONFIG}" ] && [ -d "${LIBERO_ROOT}" ]; then
    mkdir -p "$(dirname "${LIBERO_CONFIG}")"
    cat > "${LIBERO_CONFIG}" <<EOF
assets: ${LIBERO_ROOT}/./assets
bddl_files: ${LIBERO_ROOT}/./bddl_files
benchmark_root: ${LIBERO_ROOT}
datasets: ${LIBERO_ROOT}/../datasets
init_states: ${LIBERO_ROOT}/./init_files
EOF
fi

exec "$@"
