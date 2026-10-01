"""Verify every transferred byte-identical Bpaper file against the published export manifest."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "docs/source_manifest.json").read_text())
assert manifest["source_commit"] == "2cc94ff6e7c4696dc364392dc19c8137d0cb7e65"
count = 0
for item in manifest["files"]:
    path = root / item["path"]
    assert path.is_file(), f"Missing evidence or code: {item['path']}"
    raw = path.read_bytes()
    git_sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    assert git_sha == item["git_blob_sha"], f"Frozen evidence changed: {item['path']}"
    assert len(raw) == item["size_bytes"], f"Size mismatch: {item['path']}"
    count += 1

required = [
    "results/final/rows.csv",
    "results/recent_baselines/rows.csv",
    "results/recent_privic_thinning/rows.csv",
    "results/recent_comparison/matched_utility.json",
    "results/robust_informed/rows.csv",
    "results/robust_boundary/rows.csv",
    "results/data_driven_extension/rows.csv",
    "results/tmc_reviewer_stress/rows.csv",
    "results/reset_sensitivity_postfix/rows.csv",
]
assert all((root / p).is_file() for p in required)
print(f"Verified {count} byte-identical source artifacts and all {len(required)} essential result paths")
