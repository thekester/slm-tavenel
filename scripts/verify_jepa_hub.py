"""Verify a pinned Hub export in a fresh process without importing project code."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",required=True)
    parser.add_argument("--revision",required=True)
    parser.add_argument("--local-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--anonymous",action="store_true")
    args=parser.parse_args()
    if args.repo!="athenista/slm-tavenel-jepa-124m" or len(args.revision)!=40:
        raise ValueError("Expected owner-authorized repository and full revision")
    torch.set_num_threads(6)
    runtime=["model.safetensors","config.json","hf_config.py","hf_model.py","hf_representation.py",
             "tokenizer.json","tokenizer_config.json","special_tokens_map.json","package-manifest.json","verification.json"]
    folder=Path(snapshot_download(args.repo,revision=args.revision,local_dir=args.local_dir,
                                  allow_patterns=runtime,token=False if args.anonymous else None))
    manifest=json.loads((folder/"package-manifest.json").read_text(encoding="utf-8"))
    for name in runtime:
        if name=="package-manifest.json": continue
        digest=hashlib.sha256((folder/name).read_bytes()).hexdigest()
        if digest!=manifest["files"][name]:raise ValueError(f"Hub runtime differs from manifest: {name}")
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    model=AutoModel.from_pretrained(folder,local_files_only=True,trust_remote_code=True,dtype=torch.float32).eval()
    inputs=tok(["A dog runs across a grassy field.","A train is waiting at a station."],
               add_special_tokens=False,padding=True,truncation=True,max_length=64,return_tensors="pt")
    with torch.inference_mode(): output=model(**inputs).pooler_output
    parameters=sum(p.numel() for p in model.parameters())
    if list(output.shape)!=[2,576] or not torch.isfinite(output).all() or parameters!=124038144 or model.can_generate():
        raise ValueError("Unexpected shape, values, count or task")
    if any(name=="slm" or name.startswith("slm.") for name in sys.modules):
        raise ValueError("Standalone loading imported local project code")
    record={"repo":args.repo,"revision":args.revision,"anonymous":args.anonymous,"standalone_without_slm_import":True,
            "shape":list(output.shape),"parameters":parameters,"weights_sha256":manifest["files"]["model.safetensors"],
            "runtime_files_verified":len(runtime)-1,"finite_outputs":True,"generation_supported":False}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(record,indent=2),flush=True)


if __name__=="__main__": main()
