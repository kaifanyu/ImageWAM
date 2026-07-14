# syntax=docker/dockerfile:1.7
#
# ImageWAM main environment (reproduces ./.venv).
#
# Contains: FLUX.2 / OmniGen2 training + inference, and the LIBERO evaluation
# stack (mujoco 3.3.2 / robosuite 1.4.0 / bddl).  It does NOT contain the
# RoboTwin stack -- that lives in .venv-robotwin and needs SAPIEN + Vulkan.
#
# The venv is built at /opt/imagewam/.venv, deliberately OUTSIDE the repo, so
# that bind-mounting the host repo over /workspace/ImageWAM at runtime does not
# shadow it with the host's ./.venv.

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# build-essential + the devel CUDA image give us nvcc, which deepspeed needs to
# JIT its fused ops.  The mesa/osmesa/glfw trio is what scripts/setup/_install_libero_env.sh
# installs for MuJoCo offscreen rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
        libosmesa6-dev \
        libgl1-mesa-glx \
        libglfw3 \
        libglew-dev \
        libglib2.0-0 \
        patchelf \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# UV_PROJECT_ENVIRONMENT steers `uv sync`; bare `uv pip install` ignores it and
# looks for VIRTUAL_ENV instead. Both must point at the same venv or the LIBERO
# and transformers steps below install into nothing.
ENV UV_PROJECT_ENVIRONMENT=/opt/imagewam/.venv \
    VIRTUAL_ENV=/opt/imagewam/.venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    CUDA_HOME=/usr/local/cuda

RUN uv python install 3.11

WORKDIR /workspace/ImageWAM

# Resolve dependencies from the lockfile before copying source, so that editing
# src/ does not invalidate the (very expensive) torch cu118 download layer.
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra shared

# pyproject pins transformers==4.51.3 (OmniGen2 / Ovis), but the live .venv runs
# 4.56.1 because .env.local is configured for FLUX.2.  Build with
# --build-arg TRANSFORMERS_VERSION=4.51.3 to get an OmniGen2/Ovis image instead.
ARG TRANSFORMERS_VERSION=4.56.1
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "transformers==${TRANSFORMERS_VERSION}"

# --- LIBERO evaluation stack (mirrors scripts/setup/_install_libero_env.sh) ---
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        mujoco==3.3.2 \
        robosuite==1.4.0 \
        bddl==1.0.1 \
        gym==0.25.2 \
        easydict \
        thop \
        future \
        cloudpickle \
        opencv-python-headless

# LIBERO is installed editable against /workspace/ImageWAM/third_party/LIBERO.
# At runtime the bind mount replaces this clone with the host's already-patched
# copy at the same path, so the editable finder keeps resolving.
ARG LIBERO_REPO=https://github.com/Lifelong-Robot-Learning/LIBERO.git
RUN git clone --depth 1 "${LIBERO_REPO}" third_party/LIBERO \
    && touch third_party/LIBERO/libero/libero/__init__.py \
    && sed -i \
        's/init_states = torch.load(init_states_path)/init_states = torch.load(init_states_path, weights_only=False)/' \
        third_party/LIBERO/libero/libero/benchmark/__init__.py \
    && cd third_party/LIBERO \
    && uv pip install -e .

# robosuite nags on every import until a private macro file exists. This script
# imports robosuite, so it needs the same headless GL backend the runtime uses --
# both MUJOCO_GL and PATH are only exported in the ENV block below, so set them
# inline here or mujoco tries EGL and dies on a missing display.
RUN MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa /opt/imagewam/.venv/bin/python \
        /opt/imagewam/.venv/lib/python3.11/site-packages/robosuite/scripts/setup_macros.py

# The container runs as your host uid so bind-mounted files stay yours, but an
# unknown uid has no /etc/passwd entry -- and getpass.getuser() then raises
# KeyError, which is enough to make `import deepspeed` fail (torch's inductor
# cache-dir helper calls it at import time). Give the uid a real account.
ARG UID=3041
ARG GID=3041
RUN groupadd -g "${GID}" imagewam \
    && useradd -u "${UID}" -g "${GID}" -m -d /home/imagewam -s /bin/bash imagewam

ENV PATH=/opt/imagewam/.venv/bin:$PATH \
    PYTHON_BIN=/opt/imagewam/.venv/bin/python \
    PYTHONPATH=/workspace/ImageWAM/third_party/LIBERO \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SAPIEN/RoboTwin needs graphics; LIBERO does not, but requesting the capability
# is harmless and keeps the image usable if you later add the RoboTwin stage.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
# Strip CRs defensively: a Windows checkout (core.autocrlf) yields a CRLF script,
# and the kernel then fails to exec "#!/usr/bin/env bash\r". .gitattributes should
# prevent this, but the image must not depend on the host's git config.
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
