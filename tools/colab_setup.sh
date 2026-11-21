#!/usr/bin/env bash
# One-call Colab setup: sync code from GitHub, install deps, restore data and weights from Drive.
# Safe to re-run; each step is skipped when already done.
#
#   bash tools/colab_setup.sh <branch> <drive_root>
#
# Expected Drive layout (upload once):
#   <drive_root>/data/images.tar.gz                 official Oxford-IIIT Pet images archive
#   <drive_root>/data/coco/{train,val,test}.json    from tools/pets_to_coco.py (seed 0)
#   <drive_root>/data/coco/split_manifest.txt
#   <drive_root>/weights/deim_dfine_hgnetv2_n_coco_160e.pth
#   <drive_root>/outputs/                           run outputs are written here
set -euo pipefail

BRANCH="${1:-CSI5341-Project}"
DRIVE_ROOT="${2:-/content/drive/MyDrive/CSI5341-DEIM}"
REPO_URL="https://github.com/AliAdenAwil/CSI5341-DEIM.git"
REPO_DIR="/content/CSI5341-DEIM"

# 1. Code: always match the pushed branch exactly (local edits made inside Colab are discarded).
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone -q -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch -q origin "$BRANCH"
git checkout -q -B "$BRANCH" "origin/$BRANCH"
git reset -q --hard "origin/$BRANCH"
echo "code: $(git log -1 --format='%h %s')"

# 2. Dependencies: keep Colab's preinstalled CUDA torch/torchvision.
if [ ! -f /content/.deim_deps_ok ]; then
  grep -v -E '^(torch|torchvision)([<>=]|$)' requirements.txt | pip install -q -r /dev/stdin
  touch /content/.deim_deps_ok
fi

# 3. Data: images are unpacked to the local disk (reading 7k files from Drive is slow).
mkdir -p data/oxford_pets
if [ ! -d data/oxford_pets/images ] || [ "$(ls data/oxford_pets/images | wc -l)" -lt 7000 ]; then
  echo "unpacking images..."
  tar -xzf "$DRIVE_ROOT/data/images.tar.gz" -C data/oxford_pets
fi
rm -rf data/oxford_pets/coco
cp -r "$DRIVE_ROOT/data/coco" data/oxford_pets/coco
python - <<'EOF'
import json
n = {s: len(json.load(open(f'data/oxford_pets/coco/{s}.json'))['images']) for s in ('train', 'val', 'test')}
assert n == {'train': 2937, 'val': 367, 'test': 367}, n
print('data:', n)
EOF

# 4. Weights and outputs live on Drive.
mkdir -p weights "$DRIVE_ROOT/outputs" "$DRIVE_ROOT/weights/hgnetv2"
ln -sfn "$DRIVE_ROOT/weights/deim_dfine_hgnetv2_n_coco_160e.pth" weights/deim_dfine_hgnetv2_n_coco_160e.pth
ln -sfn "$DRIVE_ROOT/outputs" outputs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARNING: no GPU attached"
echo "setup done"
