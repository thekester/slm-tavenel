"""Check the recorded N layer-0 result with unbatched legacy arithmetic, without unused later layers."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from slm.runlog import sha256_file, write_json
from slm.train import load_model
import audit_exp0046 as audit


def main():
    from tokenizers import Tokenizer
    torch.set_num_threads(6)
    out = ROOT / "runs/EXP-0046-audit-2026-11-11"
    checkpoint = ROOT / "runs/EXP-0046-124m-ntp/checkpoints/step_00009000.pt"
    rows = json.loads((audit.DATA / "val.json").read_text(encoding="utf-8"))
    model = load_model(checkpoint, "cpu").eval()
    tok = Tokenizer.from_file(str(audit.evaluator.TOKENIZER))
    features = []
    with torch.no_grad():
        for row in rows:
            ids = torch.tensor([tok.encode(row["caption"]).ids[:64]])
            h = model.hook_embed(model.embed(ids))
            h = model.blocks[0](h, model.rope_cos, model.rope_sin, 0, None, None)
            features.append(h[0].float().mean(0).numpy())
    features = np.stack(features)
    anchors = [0, 999, 2499, 2500, 3999, 4999]
    legacy = audit.evaluator.native_layers(model, [rows[i]["caption"] for i in anchors])[0]
    np.testing.assert_allclose(features[anchors], legacy, rtol=0, atol=0)
    np.save(out / "N-legacy-first-block.npy", features)
    image = np.load(ROOT / "runs/EXP-0046-evaluations/dinov2-val2017.npy")
    idx = np.random.default_rng(20261109).permutation(5000)[2500:]
    items = audit.evaluator.mutual_knn_items(features[idx], image[idx])
    original = json.loads((ROOT / "runs/EXP-0046-evaluations/text-image-alignment.json").read_text())
    result = {"checkpoint_sha256": sha256_file(checkpoint), "val_rows_sha256": sha256_file(audit.DATA / "val.json"),
              "script_sha256": sha256_file(__file__), "selected_layer_fixed_from_original": 0,
              "six_legacy_anchor_vectors_bitwise_equal": True, "primary_mean_unbatched": float(items.mean()),
              "original_rounded_primary": original["results"]["N"]["mutual_knn10_primary_heldout"],
              "features_sha256": sha256_file(out / "N-legacy-first-block.npy")}
    write_json(out / "N-first-layer-check.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
