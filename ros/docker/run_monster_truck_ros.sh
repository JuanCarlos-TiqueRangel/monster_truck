#!/usr/bin/env bash
set -e

IMAGE_NAME="monster_truck_ros:v1"
CONTAINER_NAME="monster_truck_ros"
DOCKERFILE="monster_truck.Dockerfile"

FORCE_BUILD=false

# Optional: pass --build to force rebuild
if [[ "$1" == "--build" ]]; then
  FORCE_BUILD=true
fi

# Build image only if:
#  - it does not exist, or
#  - user explicitly asked for --build
if $FORCE_BUILD || ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "[*] Building image ${IMAGE_NAME}..."
  docker build -t "${IMAGE_NAME}" -f "${DOCKERFILE}" .
else
  echo "[*] Image ${IMAGE_NAME} already exists, skipping build."
fi

# Remove old container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
  echo "[*] Removing existing container ${CONTAINER_NAME}..."
  docker rm -f "${CONTAINER_NAME}"
fi

PROJECT_ROOT="$(cd .. && pwd)"

echo "[*] Starting container ${CONTAINER_NAME} from image ${IMAGE_NAME}..."
docker run -it \
  --gpus all \
  --runtime=nvidia \
  --name "${CONTAINER_NAME}" \
  --network host \
  -v "${PROJECT_ROOT}:/ros_ws" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -e DISPLAY="${DISPLAY}" \
  -w /ros_ws/src \
  "${IMAGE_NAME}" \
  bash


