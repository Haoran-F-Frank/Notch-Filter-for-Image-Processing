#!/bin/bash
# Apply FoundDiff-tools to the repo root (run from ~/FoundDiff2.0/FoundDiff).
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="$(cd "$(dirname "$0")" && pwd)"

echo "FoundDiff root: $ROOT"
echo "Tools dir:      $TOOLS"

cp "$TOOLS/demo_denoise.py" "$ROOT/"
cp "$TOOLS/denoise_nii.py" "$ROOT/"
cp "$TOOLS/convert_nii_to_npy.py" "$ROOT/"

# Patch src/DADiff.py for DA-CLIP.pth fallback
DADIFF="$ROOT/src/DADiff.py"
if grep -q "DA-CLIP.pth" "$DADIFF" 2>/dev/null; then
  echo "src/DADiff.py already patched."
else
  cp "$DADIFF" "$DADIFF.bak.$(date +%s)"
  cd "$ROOT"
  python3 - <<'PY'
from pathlib import Path
p = Path("src/DADiff.py")
text = p.read_text()
old = "            state_dict=torch.load('Dose-CLIP.pth', map_location='cpu')\n            clipiqa.load_state_dict(state_dict, strict=True)"
new = """            clip_path = 'Dose-CLIP.pth' if os.path.exists('Dose-CLIP.pth') else 'DA-CLIP.pth'
            if not os.path.exists(clip_path):
                raise FileNotFoundError(
                    'DA-CLIP weights not found. Place Dose-CLIP.pth or DA-CLIP.pth in the repo root.'
                )
            state_dict=torch.load(clip_path, map_location='cpu')
            clipiqa.load_state_dict(state_dict, strict=True)"""
if old not in text:
    raise SystemExit("ERROR: src/DADiff.py layout changed; apply patch manually.")
p.write_text(text.replace(old, new, 1))
print("Patched src/DADiff.py")
PY
fi

echo "Done. New files:"
ls -la "$ROOT/demo_denoise.py" "$ROOT/denoise_nii.py" "$ROOT/convert_nii_to_npy.py"
