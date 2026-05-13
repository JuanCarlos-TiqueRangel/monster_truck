FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# For headless MuJoCo rendering
ENV MUJOCO_GL=osmesa

# --- system deps for pip + MuJoCo rendering on Ubuntu 24.04 ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
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

# --- install Python packages ---
# psutil is installed by Debian, so we ignore the existing system copy
RUN python3 -m pip install --no-cache-dir \
    --ignore-installed psutil \
    gpytorch \
    matplotlib \
    ipympl \
    ipywidgets \
    pyserial \
    casadi \
    pymavlink \
    mujoco

# --- ROS env + workspace ---
WORKDIR /ros_ws

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc