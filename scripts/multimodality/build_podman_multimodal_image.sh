#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG=${IMAGE_TAG:-localhost/verl-multimodality:vllm012-arm64}
DOCKERFILE=${DOCKERFILE:-docker/Dockerfile.multimodality.vllm012.arm64}

cd "$(dirname "$0")/../.."
echo "Building ${IMAGE_TAG} from ${DOCKERFILE}"
podman-hpc build --pull=never -t "${IMAGE_TAG}" -f "${DOCKERFILE}" .
podman-hpc images | grep -E 'verl-multimodality|REPOSITORY' || true
