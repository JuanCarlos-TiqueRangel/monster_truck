FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Prevent JAX from preallocating most GPU memory.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Ubuntu packages and ESP-IDF prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3-pip \
    python3-venv \
    nano \
    git \
    wget \
    flex \
    bison \
    gperf \
    cmake \
    ninja-build \
    ccache \
    libffi-dev \
    libssl-dev \
    dfu-util \
    libusb-1.0-0 \
    libglfw3 \
    libglew2.2 \
    libgl1 \
    libglx-mesa0 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# Add the Espressif EIM repository BEFORE installing eim-cli
RUN echo "deb [trusted=yes] https://dl.espressif.com/dl/eim/apt/ stable main" \
        > /etc/apt/sources.list.d/espressif.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends eim-cli \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA PyTorch
RUN python3 -m pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Install JAX with CUDA 12 support
RUN python3 -m pip install --no-cache-dir \
    --ignore-installed \
    "jax[cuda12]"

# Install other Python packages
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

# ROS workspace
WORKDIR /ros_ws

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
