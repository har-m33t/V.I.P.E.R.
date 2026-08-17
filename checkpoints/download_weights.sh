#!/usr/bin/env bash
# Fetch VIPER model weights from the GitHub Release and verify them.
#
# Weights are Release assets rather than committed files because every
# checkpoint exceeds GitHub's hard 100 MB per-file limit.
#
#   ./checkpoints/download_weights.sh              # all available weights
#   ./checkpoints/download_weights.sh best_model.pth viper_convnext_unfrozen_lr1e4.pt
set -euo pipefail

REPO="${VIPER_REPO:-har-m33t/V.I.P.E.R.}"
TAG="${VIPER_WEIGHTS_TAG:-weights-v1}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="https://github.com/${REPO}/releases/download/${TAG}"

ALL=(
  best_model.pth
  unfrozen_linear_lr1e4.pt
  viper_convnext_unfrozen_lr1e4.pt
  viper_convnext_unfrozen_lr1e4_patience4.pt
  convnext_cfsmall_v1.pt
  probe_clean_v1.pt probe_clean_v2.pt
  probe_blur3_v1.pt probe_blur3_v2.pt
)
FILES=("$@")
[ ${#FILES[@]} -eq 0 ] && FILES=("${ALL[@]}")

for f in "${FILES[@]}"; do
  if [ -f "$DEST/$f" ]; then
    echo "have    $f"
    continue
  fi
  echo "fetch   $f"
  curl -fL --progress-bar -o "$DEST/$f.part" "$BASE/$f"
  mv "$DEST/$f.part" "$DEST/$f"
done

# --ignore-missing so verifying a subset does not fail on the others.
echo
if command -v shasum >/dev/null 2>&1; then
  ( cd "$DEST" && shasum -a 256 -c SHA256SUMS --ignore-missing )
elif command -v sha256sum >/dev/null 2>&1; then
  ( cd "$DEST" && sha256sum -c SHA256SUMS --ignore-missing )
else
  echo "no shasum/sha256sum available - skipping verification" >&2
fi
