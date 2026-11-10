"""Reconcile the legacy automatic and explicit slow tiny1m tokenizers on the old dev corpus."""
from pathlib import Path
import sys
import json

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from slm.evaluation import bits_per_byte, read_texts
from slm.hf_adapter import HFModel, HFTokenizer
from slm.runlog import sha256_file


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
    torch.set_num_threads(2)
    folder=ROOT/'data/models/fluency/tiny1m/hf'
    data=ROOT/'data/processed/exp0045-simple-stories-v2/dev.jsonl'
    old=ROOT/'runs/EXP-0045/evaluations-100m-v2/tiny1m-dev-bpb.json'
    fast=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    slow=LlamaTokenizer(vocab_file=str(folder/'tokenizer.model'),legacy=True)
    fast.model_max_length=slow.model_max_length=2**31-1
    texts=read_texts(data)
    a=[fast.encode(t,add_special_tokens=False) for t in texts]
    b=[slow.encode(t,add_special_tokens=False) for t in texts]
    previous=json.loads(old.read_text())
    print('Automatic tokenizer tokens:',sum(map(len,a)),'; explicit slow tokens:',sum(map(len,b)),flush=True)
    assert sum(map(len,a))==previous['tokens'], 'The automatic tokenizer does not reproduce the old token count'
    expected='f30ae85c53797b5944da7f235162756b31987233b44eac36ed449afd1b82523b'
    assert sha256_file(folder/'model.safetensors')==expected
    model=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,dtype=torch.float32).eval()
    score=bits_per_byte(HFModel(model,256),HFTokenizer(slow),texts,torch.device('cpu'),max_len=256)
    report={'kind':'slm.story_reference_tokenizer_reconciliation','reference_repo':'shibatch/tiny1m',
            'revision':'fdacad8ff9dc27bd72c77d3684075a494be1ba11','weights_sha256':expected,
            'data_sha256':sha256_file(data),'documents':len(texts),'context':256,
            'automatic_class':type(fast).__name__,'automatic_tokens':sum(map(len,a)),
            'matched_class':type(slow).__name__,'matched_legacy':True,'matched_tokens':sum(map(len,b)),
            'documents_with_different_token_ids':sum(x!=y for x,y in zip(a,b)),
            'previous_automatic_score':previous,'matched_slow_tokenizer_score':score,
            'script_sha256':sha256_file(Path(__file__)),
            'scope':'Same frozen weights and dev texts; loading difference disclosed rather than presented as model improvement'}
    out=ROOT/'runs/EXP-0045/tinystories-validation-2026-11-10/simple-dev-reference-reconciliation.json'
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
