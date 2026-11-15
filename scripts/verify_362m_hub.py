"""Download a full pinned 362M package and verify it without the local slm module."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--local-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--anonymous', action='store_true')
    args = parser.parse_args()
    if len(args.revision) != 40:
        raise ValueError('Full Hub commit required')
    repo = 'athenista/slm-tavenel-362m'
    torch.set_num_threads(6)
    folder = Path(snapshot_download(repo, revision=args.revision, local_dir=args.local_dir,
                                    token=False if args.anonymous else None))
    manifest = json.loads((folder/'package-manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or digest(path) != expected:
            raise ValueError(f'Invalid package entry: {name}')
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True,
                                                trust_remote_code=True, dtype=torch.float32).eval()
    samples = json.loads((folder/'examples.json').read_text(encoding='utf-8'))
    with torch.inference_mode():
        for sample in samples:
            inputs = tokenizer(sample['prompt'], return_tensors='pt', add_special_tokens=False)
            length = inputs['input_ids'].shape[1]
            logits = model(**inputs).logits
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite output')
            output = model.generate(**inputs, max_new_tokens=sample['max_new_tokens'],
                                    do_sample=False, use_cache=True)
            if output[0, length:].tolist() != sample['new_token_ids']:
                raise ValueError('Standalone generation does not match the package probes')
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 362286016 or not model.can_generate():
        raise ValueError('Wrong model count or task')
    if any(name == 'slm' or name.startswith('slm.') for name in sys.modules):
        raise ValueError('Standalone test imported local project code')
    record = {'repo': repo, 'revision': args.revision, 'anonymous': args.anonymous,
              'package_files_verified': len(manifest['files']),
              'weights_sha256': manifest['files']['model.safetensors'], 'parameters': parameters,
              'standalone_without_slm_import': True, 'fixed_probe_generation_matches': len(samples),
              'finite_logits': True, 'generation_supported': True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
