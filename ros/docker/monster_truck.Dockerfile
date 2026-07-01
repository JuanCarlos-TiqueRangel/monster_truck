
FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Optional but useful for JAX GPU memory behavior.
# This prevents JAX from preallocating almost all GPU memory.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# --- system deps for pip + MuJoCo rendering on Ubuntu 24.04 ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3.12-venv \
    nano \
    libglfw3 \
    libglew2.2 \
    libgl1 \
    libglx-mesa0 \
    libosmesa6 \
 && rm -rf /var/lib/apt/lists/*

# --- install CUDA PyTorch FIRST ---
RUN python3 -m pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# --- install JAX with CUDA 12 GPU support ---
RUN python3 -m pip install --no-cache-dir \
    --break-system-packages \
    --ignore-installed \
    "jax[cuda12]"

# --- install Python packages ---
RUN python3 -m pip install --no-cache-dir \
    --ignore-installed psutil \
    gpytorch \
    matplotlib \
    ipympl \
    ipywidgets \
    pyserial \
    casadi \
    pymavlink \
    mujoco \
    numba \
    pandas

# --- ROS env + workspace ---
WORKDIR /ros_ws

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc