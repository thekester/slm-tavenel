"""Prepare and verify the owner-requested JEPA Hub package on CPU; no upload."""
from __future__ import annotations

from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request

import numpy as np
import torch
from safetensors.torch import save_file, load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slm.runlog import code_revision, environment_info, sha256_file, write_json
from slm.train import load_model

OUT = ROOT / "data/models/slm-tavenel-jepa-124m"
DOCS = ROOT / "docs/model-cards/slm-tavenel-jepa-124m"
AUDIT = ROOT / "docs/reviews/exp0046-audit-2026-11-11"
CHECKPOINT = ROOT / "runs/EXP-0046-124m-jepa/checkpoints/step_00009000.pt"
TOKENIZER = ROOT / "data/tokenizers/smollm2-f8027fd0"
REPO = "athenista/slm-tavenel-jepa-124m"
EXPECTED_CHECKPOINT = "6c86f18d5fcaab3aebdc5d79e75d074a365e4edd1dd816d8689f827b773e9d43"


def copy(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set(xlim=(0, 12), ylim=(0, 5)); ax.axis("off")
    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",facecolor=color,edgecolor="#b7c6d5"))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10)
    def arrow(a,b): ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"->","color":"#45647b","lw":1.8})
    box(.15,3,2.2,1.25,"Human COCO captions\nTwo views / photo\nGroup IDs, no pixels", "#e5f1ff")
    box(3,3,3.2,1.25,"Shared causal backbone\n27 blocks, width 576\nGQA + QK norm + output gates", "#e5f1ff")
    box(7,3,2.2,1.25,"Final RMS norm\nMasked mean\nLinear 576 to 256", "#e6f5ed")
    box(9.8,3,1.95,1.25,"VICReg-style loss\nMatch paired views\nPenalize collapse", "#e6f5ed")
    arrow((2.45,3.6),(2.95,3.6)); arrow((6.3,3.6),(6.95,3.6)); arrow((9.3,3.6),(9.75,3.6))
    box(3,.55,3.2,1.25,"Inference: block 23 output\nBefore final RMS norm\nMasked mean over valid tokens", "#fff0d5")
    box(7,.55,2.2,1.25,"576-dimensional vector\nAutoModel API\nRaw, not normalized", "#fff0d5")
    arrow((4.6,2.95),(4.6,1.9)); arrow((6.3,1.2),(6.95,1.2))
    ax.text(.15,4.65,"SLM-Tavenel JEPA 124M",fontsize=18,weight="bold",color="#163c55")
    ax.text(.15,.05,"124,038,144 backbone parameters. Training projector: 147,456 separate parameters. Selected layer is zero-based.",fontsize=9,color="#45647b")
    fig.tight_layout(); fig.savefig(folder / "architecture.svg",bbox_inches="tight"); plt.close(fig)
    original = json.loads((AUDIT / "observed-original-text-image-alignment.json").read_text())
    names = {"J":"SLM-Tavenel JEPA 124M (COCO)","N":"SLM-Tavenel next-token 124M (COCO)",
             "web-124M":"SLM-Tavenel 124M (web)","random-124M":"Random backbone 124M"}
    rows = [(names.get(k,k),v["mutual_knn10_primary_heldout"],k) for k,v in original["results"].items()]
    fresh = json.loads((AUDIT / "report.json").read_text())
    rows = [(n,fresh["results"][k]["primary_mean"] if k in ("J","N") else s,k) for n,s,k in rows]
    rows.sort(key=lambda r:r[1])
    fig, ax = plt.subplots(figsize=(10.8,6))
    ax.barh([r[0] for r in rows],[r[1] for r in rows],color=["#0d9477" if r[2]=="J" else "#3788bc" if r[2]=="N" else "#aabfce" for r in rows])
    for i,(_,score,_) in enumerate(rows): ax.text(score+.002,i,f"{score:.4f}",va="center",fontsize=9)
    ax.set_xlim(0,.21); ax.set_xlabel("Shared 10-neighbour overlap with DINOv2 (higher is better)")
    ax.set_title("One metric on 2,500 held-out COCO images",loc="left",weight="bold")
    fig.text(.02,.015,"J/N: fresh audit. Public references: original evaluation. Training budgets vary; one seed; no 72B comparison.",fontsize=9,color="#45647b")
    fig.tight_layout(rect=(0,.035,1,1)); fig.savefig(folder / "neighbour-overlap.svg",bbox_inches="tight"); plt.close(fig)
    logs = [json.loads(l) for l in (AUDIT / "J-training-metrics.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(1,2,figsize=(10.8,3.5))
    axes[0].plot([r["step"] for r in logs],[r["loss"] for r in logs],color="#0d9477")
    axes[0].set(title="J: VICReg-style training objective",xlabel="Update",ylabel="Objective (not NTP cross-entropy)")
    axes[1].plot([r["step"] for r in logs],[r["gv_std_mean"] for r in logs],color="#3788bc")
    axes[1].axhline(.1,color="#ae5b40",linestyle="--",label="Recorded collapse threshold")
    axes[1].set(title="Projected representation spread",xlabel="Update",ylabel="Mean projected standard deviation")
    axes[1].legend(fontsize=8,frameon=False)
    fig.tight_layout(); fig.savefig(folder / "training.svg",bbox_inches="tight"); plt.close(fig)


def main():
    from huggingface_hub import hf_hub_download, ModelCard
    from transformers import AutoModel, AutoTokenizer
    from tokenizers import Tokenizer
    torch.set_num_threads(6)
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise ValueError("Audited checkpoint hash changed")
    OUT.mkdir(parents=True, exist_ok=True)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = state["model_config"]
    if state["step"] != 9000 or config.get("conv_layers") or not config["tie_embeddings"]:
        raise ValueError("Unexpected backbone/checkpoint configuration")
    weights = {k:v.detach().cpu().contiguous().clone() for k,v in state["model"].items()
               if not k.startswith("jepa_predictor.") and k != "lm_head.weight"}
    if any(v.dtype != torch.float32 for v in weights.values()) or sum(v.numel() for v in weights.values()) != 124038144:
        raise ValueError("Expected exact float32 backbone tensors")
    save_file(weights, str(OUT / "model.safetensors"), metadata={"format":"pt"})
    projector = {k.removeprefix("jepa_predictor."):v.detach().cpu().contiguous().clone()
                 for k,v in state["model"].items() if k.startswith("jepa_predictor.")}
    save_file(projector, str(OUT / "training_projection.safetensors"), metadata={"format":"pt"})
    del state, weights, projector
    gc.collect()
    write_json(OUT / "config.json", {**config, "model_type":"slm_tavenel", "architectures":["SlmTavenelForRepresentation"],
                "auto_map":{"AutoConfig":"hf_config.SlmTavenelConfig","AutoModel":"hf_representation.SlmTavenelForRepresentation"},
                "representation_layer":23,"representation_max_length":64,"torch_dtype":"float32",
                "use_cache":False,"pad_token_id":0,"eos_token_id":0,"bos_token_id":None})
    for name in ("hf_config.py","hf_model.py","hf_representation.py"):
        copy(ROOT / "src/slm" / name, OUT / name)
    for name in ("tokenizer.json","tokenizer_config.json","special_tokens_map.json"):
        copy(TOKENIZER / name, OUT / name)
    token_cfg = json.loads((OUT / "tokenizer_config.json").read_text(encoding="utf-8"))
    token_cfg.update(model_max_length=64,padding_side="right",pad_token="<|endoftext|>",model_input_names=["input_ids","attention_mask"])
    write_json(OUT / "tokenizer_config.json", token_cfg)
    special = json.loads((OUT / "special_tokens_map.json").read_text(encoding="utf-8"))
    special["pad_token"] = "<|endoftext|>"
    write_json(OUT / "special_tokens_map.json", special)
    if sha256_file(OUT / "tokenizer.json") != "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c":
        raise ValueError("Tokenizer mapping changed")
    copy(DOCS / "README.md", OUT / "README.md")
    copy(ROOT / "LICENSE", OUT / "LICENSE")
    for file in AUDIT.iterdir():
        if file.is_file(): copy(file, OUT / "audit" / file.name)
    copy(AUDIT / "J-training-metrics.jsonl", OUT / "training_record/metrics.jsonl")
    copy(AUDIT / "J-feature-metadata.json", OUT / "training_record/checkpoint-metadata.json")
    copy(ROOT / "data/parallel/coco2017-captions-v1/manifest.json", OUT / "training_record/data-manifest.json")
    copy(TOKENIZER / "manifest.json", OUT / "training_record/tokenizer-source-manifest.json")
    copy(ROOT / "configs/model/slm-124m-gpt2match.yaml", OUT / "training_record/architecture.yaml")
    for src,name in ((AUDIT / "observed-original-train_text_jepa.py","train_text_jepa.py"),
                     (AUDIT / "observed-original-build_coco_caption_pairs.py","build_coco_caption_pairs.py"),
                     (ROOT / "src/slm/goodview.py","goodview.py"),
                     (ROOT / "scripts/package_jepa124m.py","package_jepa124m.py")):
        copy(src, OUT / "training_code" / name)
    upstream = hf_hub_download("HuggingFaceTB/SmolLM2-360M","README.md",revision="f8027fd0eaeea54caa13c31d31b9fdc459c38b49")
    copy(Path(upstream), OUT / "third_party/smollm2_README.md")
    license_text = urllib.request.urlopen("https://www.apache.org/licenses/LICENSE-2.0.txt", timeout=30).read()
    (OUT / "TOKENIZER-LICENSE.txt").write_bytes(license_text)
    (OUT / "requirements.txt").write_text("torch>=2.3\ntransformers==4.57.6\ntokenizers==0.22.2\nsafetensors==0.8.0\n",encoding="utf-8")
    (OUT / "NOTICE.md").write_text("# Third-party materials\n\nThe unchanged tokenizer JSON comes from HuggingFaceTB/SmolLM2-360M at f8027fd0eaeea54caa13c31d31b9fdc459c38b49, declared Apache-2.0 in its pinned model card. No SmolLM2 model weights are used. Tokenizer inference metadata was changed to a 64-token cap, right padding and a PAD alias to existing EOS token 0; vocabulary/token IDs are unchanged. No separate LICENSE/NOTICE file was available at that pinned repository root; the declared Apache-2.0 text and original model card are included.\n\nCOCO annotations are attributed to Lin et al. and Chen et al.; official annotation terms are declared CC BY 4.0. Individual Flickr image terms remain separate. No source captions or image pixels are bundled. Source rights/privacy reviews remain unsigned; this is not legal clearance.\n",encoding="utf-8")
    (OUT / "REPRODUCIBILITY.md").write_text("# Reproduce the research representation\n\nUse the source and audit at https://github.com/thekester/slm-tavenel/tree/v0.1.0-dev.41 . The raw COCO annotation hashes, tokenizer pin, all original model fields and checkpoint SHA256 are recorded in provenance.json and training_record/. The observed original training loop is included, without its optimizer state or a claim of a complete original immutable run manifest.\n\nDefault AutoModel output is the pre-final-RMS-norm block-23 token state and its valid-token mean. It exposes 576 dimensions. The training objective instead applies the backbone final RMS norm, pools, then uses the separate 256-dimensional linear projector. To inspect all block states, request output_hidden_states=True; hidden_states[0] is the token embedding and hidden_states[i+1] is pre-final-norm block i.\n\nInference caps at 64 tokens, the audit cap. Training preparation caps at 48; positional tables have 2048 positions, which is not a validated long-context capability. Use the supplied evaluation code for centering, L2 normalization, graph construction, split and descriptive rotation retrieval.\n\nAll root backbone tensors and the separate training projector retain exact native float32 values. verification.json establishes export parity on fixed probes only. The audit uses one seed and includes post-hoc controls; further training-seed, vision-encoder and upstream-exposure checks remain open.\n",encoding="utf-8")
    figures()
    print("Export written; checking native tensor and representation parity",flush=True)
    native = load_model(CHECKPOINT,"cpu").eval()
    exported = AutoModel.from_pretrained(OUT,trust_remote_code=True,local_files_only=True,dtype=torch.float32).eval()
    disk = load_file(str(OUT / "model.safetensors"))
    for key,value in disk.items():
        if not torch.equal(value, native.state_dict()[key]): raise ValueError(f"Tensor changed: {key}")
    tensor_count = len(disk)
    del disk
    tok = AutoTokenizer.from_pretrained(OUT,local_files_only=True)
    tok_native = Tokenizer.from_file(str(TOKENIZER / "tokenizer.json"))
    texts = ["A dog runs across a grassy field.","A train is waiting at a station.",
             "Two people sit at a table with food.","A small boat floats near the shore."]
    native_vectors=[]
    captured=[]
    hook=native.blocks[23].register_forward_hook(lambda m,i,o:captured.append(o))
    with torch.inference_mode():
        for text in texts:
            captured.clear()
            ids=tok_native.encode(text).ids[:64]
            if ids != tok.encode(text,add_special_tokens=False)[:64]:raise ValueError("Tokenizer prefix/postprocessor mismatch")
            native(torch.tensor([ids]))
            native_vectors.append(captured[0][0].float().mean(0))
        batch=tok(texts,add_special_tokens=False,padding=True,truncation=True,max_length=64,return_tensors="pt")
        actual=exported(**batch).pooler_output
        expected=torch.stack(native_vectors)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
    hook.remove()
    verification={"kind":"slm.jepa124m.export_verification","backbone_parameters":sum(p.numel() for p in exported.parameters()),
        "source_checkpoint_sha256":EXPECTED_CHECKPOINT,"weights_sha256":sha256_file(OUT / "model.safetensors"),
        "dtype":"float32","tensors_bitwise_identical":tensor_count,"probe_captions":texts,
        "selected_layer_zero_based":23,"embedding_shape":list(actual.shape),
        "max_abs_representation_difference":float((actual-expected).abs().max()),
        "mean_cosine_native_export":float(torch.nn.functional.cosine_similarity(actual,expected).mean()),
        "standalone_AutoModel_load":True,"runtime_generation_supported":exported.can_generate(),
        "tokenizer_ids_match":True,"environment":environment_info(),
        "scope":"Packaging parity on four fixed probes, not renewed performance validation or training replication"}
    if verification["backbone_parameters"] != 124038144 or verification["runtime_generation_supported"]:
        raise ValueError("Wrong backbone count or generation capability")
    write_json(OUT / "verification.json",verification)
    weights_hash=verification["weights_sha256"]
    write_json(OUT / "publication-decision.json",{"decision":"owner explicitly requested public Hugging Face publication",
        "request":"mais le sur hugging face","date":"2026-11-11","repo":REPO,
        "source_checkpoint_sha256":EXPECTED_CHECKPOINT,"model_safetensors_sha256":weights_hash,
        "scope":"Public research preview with limitations in the model card; not a signature of source rights/privacy review or OSI certification"})
    write_json(OUT / "provenance.json",{"source_checkpoint":str(CHECKPOINT.relative_to(ROOT)),
        "source_checkpoint_sha256":EXPECTED_CHECKPOINT,"source_step":9000,"weights_sha256":weights_hash,
        "backbone_parameters":124038144,"training_projection_parameters":147456,"model_config":config,
        "tokenizer_repo":"HuggingFaceTB/SmolLM2-360M","tokenizer_revision":"f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        "tokenizer_json_sha256":sha256_file(OUT / "tokenizer.json"),"packaging_code":code_revision(),
        "packager_sha256":sha256_file(__file__),"source_and_privacy_reviews":"unsigned; disclosed research preview",
        "original_execution_provenance":"Incomplete original custom-loop manifest; original metadata retained in training_record/"})
    ModelCard.load(OUT / "README.md").validate()
    write_json(OUT / "package-manifest.json",{"kind":"slm.jepa124m.hf_package","repo":REPO,
        "files":{str(p.relative_to(OUT)).replace('\\','/'):sha256_file(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='package-manifest.json' and '.cache' not in p.parts}})
    for name in ("verification.json","provenance.json","publication-decision.json","package-manifest.json"):
        copy(OUT / name,DOCS / name)
    print(json.dumps(verification,indent=2),flush=True)


if __name__ == "__main__":
    main()
