FROM osrf/ros:jazzy-desktop
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv nano \
    libgl1 libglx-mesa0 libosmesa6 libglfw3 libglew2.2 \
 && rm -rf /var/lib/apt/lists/*

# IMPORTANT PINS for gpflow stack stability
RUN python3 -m pip install --pre --no-cache-dir tf-nightly tfp-nightly && \
    python3 -m pip install --no-cache-dir "numpy<2" "setuptools<82" && \
    python3 -m pip install --no-deps gpflow==2.10.0 && \
    python3 -m pip install check-shapes deprecated multipledispatch scipy tabulate packaging


WORKDIR /ros_ws
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
