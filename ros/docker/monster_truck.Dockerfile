FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive
# Allow pip to install into the system Python (PEP 668 override)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

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

# --- Python packages (what you had in the notebook) ---
RUN python3 -m pip install \
        gpytorch \
        matplotlib \
        ipympl \
        ipywidgets && \
    python3 -m pip install \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -m pip install mujoco

# --- ROS env + workspace ---
WORKDIR /ros_ws
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
