#!/usr/bin/env bash
set -euo pipefail

# Rebuild/install GroundingDINO with CUDA extension support for modern Torch/CUDA
# stacks (e.g., RTX 50-series + sm_120). This script is intended for the
# Python 3.12 ROS environment used in this repository.

ENV_NAME="${1:-vlfm_ros312}"
GDINO_REF="${GDINO_REF:-main}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[install_groundingdino_cuda] conda is required but was not found." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
PYBIN="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

if [[ ! -x "${PYBIN}" ]]; then
  echo "[install_groundingdino_cuda] Python not found for env '${ENV_NAME}': ${PYBIN}" >&2
  exit 1
fi

echo "[install_groundingdino_cuda] Ensuring nvcc (CUDA 12.8) is available in ${ENV_NAME}"
conda install -n "${ENV_NAME}" -y -c nvidia cuda-nvcc=12.8

echo "[install_groundingdino_cuda] Cloning GroundingDINO (${GDINO_REF})"
BUILD_DIR="$(mktemp -d /tmp/groundingdino_build_XXXXXX)"
git clone https://github.com/IDEA-Research/GroundingDINO.git "${BUILD_DIR}"
(
  cd "${BUILD_DIR}"
  git checkout "${GDINO_REF}"
)

CUDA_SRC="${BUILD_DIR}/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu"

echo "[install_groundingdino_cuda] Patching CUDA extension source for newer torch C++ API"
sed -i 's/value.type().is_cuda()/value.is_cuda()/g' "${CUDA_SRC}"
sed -i 's/AT_DISPATCH_FLOATING_TYPES(value.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' "${CUDA_SRC}"

echo "[install_groundingdino_cuda] Building and installing patched GroundingDINO"
CUDA_HOME="${CONDA_BASE}/envs/${ENV_NAME}"
NV_SITE="${CUDA_HOME}/lib/python3.12/site-packages/nvidia"
INC="$(find "${NV_SITE}" -maxdepth 2 -type d -name include | tr '\n' ':')"
LIB="$(find "${NV_SITE}" -maxdepth 2 -type d -name lib | tr '\n' ':')"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export C_INCLUDE_PATH="${CUDA_HOME}/targets/x86_64-linux/include:${INC%:}:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/targets/x86_64-linux/include:${INC%:}:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${LIB%:}:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${LIB%:}:${LD_LIBRARY_PATH:-}"

"${PYBIN}" -m pip install --no-deps --force-reinstall --no-build-isolation "${BUILD_DIR}"

echo "[install_groundingdino_cuda] Done. Verifying extension import..."
"${PYBIN}" - <<'PY'
import groundingdino
from groundingdino import _C
print("groundingdino", groundingdino.__version__ if hasattr(groundingdino, "__version__") else "unknown")
print("extension_loaded", _C is not None)
PY

echo "[install_groundingdino_cuda] Success"
