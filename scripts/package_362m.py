"""Assemble and verify the owner-requested 362M research release; never upload."""
from __future__ import annotations

import gc
import gzip
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

import numpy as np
import torch
from huggingface_hub import ModelCard, hf_hub_download
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from slm.runlog import code_revision, environment_info, sha256_file, write_json
from slm.train import load_model

RUN = ROOT / 'runs/EXP-0034b-360m-muon'
EVAL = ROOT / 'runs/FAMILY-evaluations/360m-muon'
OUT = ROOT / 'data/models/slm-tavenel-362m'
DOCS = ROOT / 'docs/model-cards/slm-tavenel-362m'
TOKENIZER = ROOT / 'data/tokenizers/smollm2-f8027fd0'
CHECKPOINT = RUN / 'checkpoints/step_00008000.pt'
EXPECTED = '281c53a7dac5bb92d1c6e49a957bbb996123f523b88fa0f81178e8cbd08d8fed'
TOKEN_SHA = '9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c'
REPO = 'athenista/slm-tavenel-362m'
TASKS = ['arc_easy', 'arc_challenge', 'hellaswag', 'belebele_en', 'belebele_fr', 'winogrande']
LABELS = ['ARC-Easy', 'ARC-Challenge', 'HellaSwag', 'Belebele EN', 'Belebele FR', 'WinoGrande']
PROMPTS = ['The scientific method is based on', 'La recherche scientifique permet de',
           'Once upon a time, a fox found a key.']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def copy(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


def compressed(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('wb') as target:
        with gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0) as stream:
            stream.write(source.read_bytes())


def figures(reports, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
    folder = OUT / 'figures'
    folder.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axis('off'); ax.set(xlim=(0, 12), ylim=(0, 4.6))
    boxes = [(.2, 1.7, 2.1, 'Text input\n49,152-token BPE\nTied embedding', '#e3efff'),
             (3, 1.2, 5, '32 causal decoder blocks, width 960\nRMSNorm -> GQA (15 Q / 5 KV) -> sigmoid head gate\n+ residual -> RMSNorm -> SwiGLU (2,560) -> + residual\nQK normalization, RoPE 100,000', '#e5f4ed'),
             (8.8, 1.7, 2.9, 'Final RMSNorm\nTied language-model head\nNext-token logits', '#fff0d5')]
    for x, y, w, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,1.6,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#a7b7c6'))
        ax.text(x+w/2,y+.8,text,ha='center',va='center',fontsize=9)
    for a, b in [((2.4,2.5),(2.95,2.5)),((8.1,2.5),(8.75,2.5))]:
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':2,'color':'#476582'})
    ax.text(.2,4,'SLM-Tavenel 362M architecture',fontsize=19,weight='bold',color='#183a56')
    ax.text(.2,.3,'362,286,016 parameters | 2,048-token context | base model, no instruction tuning',fontsize=11)
    fig.tight_layout(); fig.savefig(folder / 'architecture.svg',bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 5.6))
    x=np.arange(len(TASKS)); width=.16
    names=['SLM-Tavenel 362M','SLM-Tavenel 124M','GPT-2 Medium 355M','Pythia 410M','SmolLM2 360M']
    colors=['#1976b9','#61b5a2','#929eaa','#dca444','#956eb0']
    for i,(name,report,color) in enumerate(zip(names,reports,colors)):
        scores=[report['results'][t]['acc']*100 for t in TASKS]
        errors=[report['results'][t]['acc_stderr']*196 for t in TASKS]
        ax.bar(x+(i-2)*width,scores,width,label=name,color=color,yerr=errors,capsize=1.8)
    ax.set_xticks(x,LABELS); ax.set_ylabel('Raw multiple-choice accuracy (%)')
    ax.set_ylim(0,82); ax.legend(loc='upper left',ncol=3,fontsize=9)
    ax.set_title('Same benchmark files and project scorer; different pretraining budgets',loc='left',weight='bold')
    fig.text(.04,.01,'Bars: raw acc. Error bars: approximate 95% item intervals (1.96 SE), not training-seed uncertainty. Context caps differ; see protocol.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(folder / 'benchmarks.svg',bbox_inches='tight'); plt.close(fig)
    metrics=[json.loads(line) for line in (RUN/'metrics.jsonl').read_text().splitlines()]
    evals=[json.loads(line) for line in (RUN/'eval.jsonl').read_text().splitlines()]
    # Keep the last logged value at each resumed step; preserve the full original logs separately.
    m={r['step']:r for r in metrics}; v={r['step']:r for r in evals}
    fig, ax=plt.subplots(figsize=(11,4))
    ax.plot(sorted(m),[m[s]['loss'] for s in sorted(m)],label='SLM-Tavenel 362M training loss',alpha=.8)
    ax.plot(sorted(v),[v[s]['val_loss'] for s in sorted(v)],label='Fixed-window validation loss',lw=2)
    ax.set(xlabel='Optimizer updates',ylabel='Cross-entropy (nats)',title='SLM-Tavenel 362M: recorded training progress')
    ax.legend(); fig.tight_layout();fig.savefig(folder/'training.svg',bbox_inches='tight');plt.close(fig)
    counts=manifest['records_by_split_language']
    fig,ax=plt.subplots(figsize=(10,4))
    x=np.arange(3); en=[counts[f'{s}_en'] for s in ['train','dev','test']]; fr=[counts[f'{s}_fr'] for s in ['train','dev','test']]
    ax.bar(x-.18,en,.36,label='English',color='#1976b9');ax.bar(x+.18,fr,.36,label='French',color='#61b5a2')
    ax.set_xticks(x,['Training pool','Development','Held-out test pool']);ax.set_yscale('log');ax.set_ylabel('Documents (log scale)')
    ax.set_title('SLM-Tavenel 362M: language and evaluation role',loc='left',weight='bold');ax.legend()
    for i,(a,b) in enumerate(zip(en,fr)):
        ax.text(i-.18,a*1.1,f'{a:,}',ha='center',fontsize=9);ax.text(i+.18,b*1.1,f'{b:,}',ha='center',fontsize=9)
    ax.set_ylim(1000,7e6);fig.tight_layout();fig.savefig(folder/'data-split.svg',bbox_inches='tight');plt.close(fig)


def main():
    torch.set_num_threads(6)
    OUT.mkdir(parents=True,exist_ok=True);DOCS.mkdir(parents=True,exist_ok=True)
    if sha256_file(CHECKPOINT)!=EXPECTED or sha256_file(TOKENIZER/'tokenizer.json')!=TOKEN_SHA:
        raise ValueError('Source checkpoint or tokenizer changed')
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=False,mmap=True)
    if checkpoint['step']!=8000 or checkpoint['model_config']['conv_layers']:
        raise ValueError('Unexpected model or step')
    state={k:v.detach().float().contiguous() for k,v in checkpoint['model'].items() if k!='lm_head.weight'}
    if not torch.equal(checkpoint['model']['embed.weight'],checkpoint['model']['lm_head.weight']):
        raise ValueError('Embedding/head alias differs')
    save_file(state,str(OUT/'model.safetensors'),metadata={'format':'pt'})
    cfg={**checkpoint['model_config'],'model_type':'slm_tavenel','architectures':['SlmTavenelForCausalLM'],
         'auto_map':{'AutoConfig':'hf_config.SlmTavenelConfig','AutoModelForCausalLM':'hf_model.SlmTavenelForCausalLM'},
         'torch_dtype':'float32','max_position_embeddings':2048,'tie_word_embeddings':True,
         'eos_token_id':0,'pad_token_id':0,'bos_token_id':None,'use_cache':True}
    write_json(OUT/'config.json',cfg)
    write_json(OUT/'generation_config.json',{'eos_token_id':0,'pad_token_id':0,'use_cache':True,'do_sample':False})
    for name in ['hf_model.py','hf_config.py']:copy(ROOT/'src/slm'/name,OUT/name)
    for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json']:copy(TOKENIZER/name,OUT/name)
    tok_cfg=read(OUT/'tokenizer_config.json')
    tok_cfg.update(model_max_length=2048,padding_side='right',pad_token='<|endoftext|>',model_input_names=['input_ids','attention_mask'])
    write_json(OUT/'tokenizer_config.json',tok_cfg)
    special=read(OUT/'special_tokens_map.json');special['pad_token']='<|endoftext|>';write_json(OUT/'special_tokens_map.json',special)
    del checkpoint,state;gc.collect()
    print('Export written; checking all tensors, native logits and generation cache on CPU',flush=True)
    native=load_model(CHECKPOINT,'cpu').eval()
    hf=AutoModelForCausalLM.from_pretrained(OUT,trust_remote_code=True,local_files_only=True,dtype=torch.float32).eval()
    disk=load_file(str(OUT/'model.safetensors'))
    for name,tensor in disk.items():
        if not torch.equal(tensor,native.state_dict()[name]):raise ValueError(f'Export differs: {name}')
    tensor_count=len(disk);del disk;gc.collect()
    tok=AutoTokenizer.from_pretrained(OUT,local_files_only=True)
    from tokenizers import Tokenizer
    source_tok=Tokenizer.from_file(str(TOKENIZER/'tokenizer.json'))
    errors=[];samples=[];cache_checks=[]
    with torch.inference_mode():
        for prompt in PROMPTS:
            inputs=tok(prompt,return_tensors='pt',add_special_tokens=False)
            ids=inputs['input_ids']
            if ids[0].tolist()!=source_tok.encode(prompt).ids:raise ValueError('Token IDs changed')
            expected=native(ids)[0];actual=hf(**inputs,use_cache=False).logits
            errors.append(float((expected-actual).abs().max()))
            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
            plain=hf.generate(**inputs,max_new_tokens=12,do_sample=False,use_cache=False)
            cached=hf.generate(**inputs,max_new_tokens=12,do_sample=False,use_cache=True)
            if not torch.equal(plain,cached):raise ValueError('Cached and uncached tokens differ')
            cache_checks.append(True)
            samples.append({'prompt':prompt,'do_sample':False,'max_new_tokens':12,
                            'new_text':tok.decode(cached[0,ids.shape[1]:],skip_special_tokens=True),
                            'new_token_ids':cached[0,ids.shape[1]:].tolist()})
    parameters=sum(p.numel() for p in hf.parameters())
    if parameters!=362286016:raise ValueError('Wrong parameter count')
    del native;gc.collect()
    # Small, explicitly defined CPU benchmark; no GPU interference or capacity extrapolation.
    ids=tok(('Scientific research relies on observations. '*50),return_tensors='pt')['input_ids'][:,:128]
    elapsed=[];counts=[]
    with torch.inference_mode():
        for iteration in range(4):
            start=time.perf_counter()
            output=hf.generate(ids,attention_mask=torch.ones_like(ids),min_new_tokens=32,max_new_tokens=32,
                               do_sample=False,use_cache=True)
            duration=time.perf_counter()-start
            if iteration:elapsed.append(duration);counts.append(int(output.shape[1]-ids.shape[1]))
    verification={'checkpoint_sha256':EXPECTED,'weights_sha256':sha256_file(OUT/'model.safetensors'),
                  'parameters':parameters,'export_dtype':'float32','all_exported_tensors_match_native':True,
                  'exported_tensor_count':tensor_count,'lm_head_tied_to_embedding':hf.lm_head.weight is hf.embed.weight,
                  'max_abs_logit_error_fixed_probes':max(errors),'cached_uncached_greedy_tokens_identical':cache_checks,
                  'tokenizer_json_unchanged':sha256_file(OUT/'tokenizer.json')==TOKEN_SHA,
                  'cpu_inference':{'threads':6,'prompt_tokens':128,'new_tokens':counts,'timed_repeats':3,'warmups':1,
                                   'dtype':'float32','kv_cache':True,'batch_size':1,'seconds':elapsed,
                                   'median_tokens_per_second':32/statistics.median(elapsed)},
                  'environment':environment_info(),'scope':'Fixed-probe export parity and short CPU timing; not a fresh six-task evaluation or complete safety review'}
    write_json(OUT/'verification.json',verification);write_json(OUT/'examples.json',samples)
    del hf;gc.collect()
    print(json.dumps({k:verification[k] for k in ['parameters','max_abs_logit_error_fixed_probes','cpu_inference']},indent=2),flush=True)
    for name in ['metrics.jsonl','eval.jsonl','observability.jsonl','run_manifest.json']:
        copy(RUN/name,OUT/'training_record'/name)
    for path in RUN.glob('energy_meter_*.json'):copy(path,OUT/'training_record'/path.name)
    for src,name in [(ROOT/'configs/model/slm-360m-classic.yaml','architecture.yaml'),
                     (ROOT/'configs/train/exp0034b-360m-muon.yaml','recipe.yaml'),
                     (ROOT/'data/processed/slm-tavenel-edu-corpus-v2/manifest.json','corpus-manifest.json'),
                     (ROOT/'data/tokens/slm-tavenel-edu-corpus-v2-smollm2/manifest.json','tokens-manifest.json'),
                     (TOKENIZER/'manifest.json','tokenizer-source-manifest.json')]:copy(src,OUT/'training_record'/name)
    for src in [ROOT/'src/slm/train.py',ROOT/'src/slm/model.py',ROOT/'src/slm/config.py',ROOT/'src/slm/optim.py',
                ROOT/'src/slm/evaluation.py',ROOT/'src/slm/benchmarks.py',ROOT/'scripts/build_edu_corpus_v1.py',
                ROOT/'scripts/compare_mc_paired.py',ROOT/'scripts/compare_exp0034b_vs_124m_bpb.py',Path(__file__)]:
        copy(src,OUT/'source_snapshot'/src.name)
    reports=[read(EVAL/'benchmarks.json')]
    for path in [ROOT/'runs/EXP-0036-evaluations/124m-gpt2match-muon/benchmarks.json',
                 ROOT/'runs/reference/peer-models-2026-10/gpt-2-medium/mc.json',
                 ROOT/'runs/reference/peer-models-2026-10/pythia-410m/mc.json',
                 ROOT/'runs/reference/peer-models-2026-10/smollm2-360m/mc.json']:
        reports.append(read(path))
    for i,report in enumerate(reports):
        if report['benchmarks']!=reports[0]['benchmarks']:raise ValueError('Benchmark file hashes differ')
        for task in TASKS:
            items=report['results'][task]['per_item']
            original=reports[0]['results'][task]['per_item']
            if [(x['id'],x['label']) for x in items]!=[(x['id'],x['label']) for x in original]:raise ValueError('Benchmark item alignment changed')
            if abs(sum(x['correct'] for x in items)/len(items)-report['results'][task]['acc'])>1e-12:raise ValueError('Incorrect aggregate accuracy')
        write_json(OUT/'evaluation'/f'model-{i}-metadata.json',{k:v for k,v in report.items() if k!='results'})
    for name in ['benchmarks.json','edu-test-bpb.json']:
        compressed(EVAL/name,OUT/'evaluation'/f'{name}.gz')
    for name in ['vs-124m-paired.json','vs-gpt2-medium-paired.json','vs-pythia-410m-paired.json',
                 'vs-smollm2-360m-paired.json','paired-vs-124m.json']:
        copy(EVAL/name,OUT/'evaluation'/name)
    summary={'models':[{**{k:v for k,v in report.items() if k!='results'},
                       'results':{t:{k:v for k,v in s.items() if k!='per_item'} for t,s in report['results'].items()}}
                      for report in reports],
             'note':'Existing native-checkpoint scores, checked against saved items; no full benchmark rerun during packaging.'}
    write_json(OUT/'evaluation/summary.json',summary)
    corpus=read(ROOT/'data/processed/slm-tavenel-edu-corpus-v2/manifest.json')
    figures(reports,corpus)
    for dataset,revision,name in [('HuggingFaceFW/fineweb-edu',corpus['sources']['en']['revision'],'fineweb-edu'),
                                  ('epfml/FineWeb2-HQ',corpus['sources']['fr']['revision'],'fineweb2-hq')]:
        src=hf_hub_download(dataset,'README.md',repo_type='dataset',revision=revision)
        copy(Path(src),OUT/'third_party'/f'{name}-README.md')
    for name in ['LICENSE','TOKENIZER-LICENSE.txt']:
        copy(ROOT/'LICENSE' if name=='LICENSE' else ROOT/'data/models/slm-tavenel-jepa-124m/TOKENIZER-LICENSE.txt',OUT/name)
    copy(ROOT/'data/models/slm-tavenel-jepa-124m/third_party/smollm2_README.md',OUT/'third_party/smollm2-README.md')
    (OUT/'requirements.txt').write_text('torch==2.10.0\ntransformers==4.57.6\ntokenizers==0.22.2\nsafetensors==0.8.0\n',encoding='utf-8')
    (OUT/'NOTICE.md').write_text('''# Attribution and source terms

Project code and author-owned weight contributions: MIT, Theophile Avenel and contributors.
The unchanged tokenizer.json is from HuggingFaceTB/SmolLM2-360M at f8027fd0eaeea54caa13c31d31b9fdc459c38b49, declared Apache-2.0 in its pinned model card. No SmolLM2 model weights are used. The full Apache license and upstream card are included. Inference metadata changes set a 2,048-token maximum, right padding and PAD as an alias of existing EOS token 0; token IDs and vocabulary are unchanged.

Training text: HuggingFaceFW/fineweb-edu and epfml/FineWeb2-HQ; exact revisions, file lists, declared ODC-By 1.0 compilation terms and filtering are in training_record/corpus-manifest.json and the pinned source cards in third_party/. Common Crawl and individual source-page terms remain relevant. Database licensing does not establish all underlying page rights. No training text is redistributed in this model package. Rights and manual privacy reviews remain unsigned; public research release authorization is not a legal clearance or OSI certification.
''',encoding='utf-8')
    write_json(OUT/'provenance.json',{'source_checkpoint':str(CHECKPOINT.relative_to(ROOT)),
                                    'source_checkpoint_sha256':EXPECTED,'source_step':8000,
                                    'weights_sha256':verification['weights_sha256'],'parameters':parameters,
                                    'tokenizer_json_sha256':TOKEN_SHA,'packaging_code':code_revision(),
                                    'packager_sha256':sha256_file(__file__),
                                    'source_snapshot_scope':'Current packaging-time source snapshot; training was resumed in dirty worktrees. Exact historical executed source for every segment is not available.',
                                    'checkpoint_conversion':'Checkpoint model tensors widened to float32, as native load_model does; separate optimizer master weights not exported.',
                                    'source_and_privacy_review':'unsigned; declared dataset terms and limitations disclosed'})
    write_json(OUT/'publication-decision.json',{'date':'2026-11-15','repo':REPO,
                                              'decision':'Owner requested publication of the newly completed model on Hugging Face with documented research limitations',
                                              'request':"je te laisse commit an dpush and releas et y a un ouveua modèlke à mettre sur hugging afce",
                                              'source_checkpoint_sha256':EXPECTED,
                                              'weights_sha256':verification['weights_sha256'],
                                              'scope':'Public research preview; not a signature of source-rights/privacy reviews or a claim of OSI certification'})
    copy(DOCS/'README.md',OUT/'README.md')
    ModelCard.load(OUT/'README.md').validate()
    write_json(OUT/'package-manifest.json',{'kind':'slm.362m.hf_package','repo':REPO,
                                          'files':{p.relative_to(OUT).as_posix():sha256_file(p) for p in sorted(OUT.rglob('*'))
                                                   if p.is_file() and p.name!='package-manifest.json' and '.cache' not in p.parts}})
    for name in ['verification.json','provenance.json','publication-decision.json','package-manifest.json','examples.json']:
        copy(OUT/name,DOCS/name)
    # Compact public evidence in Git; no weights or corpus text.
    for path in (OUT/'figures').glob('*.svg'):copy(path,DOCS/'figures'/path.name)
    for name in ['summary.json','vs-124m-paired.json','vs-gpt2-medium-paired.json',
                 'vs-pythia-410m-paired.json','vs-smollm2-360m-paired.json','paired-vs-124m.json']:
        copy(OUT/'evaluation'/name,DOCS/'evaluation'/name)
    print('Package complete: '+str(OUT),flush=True)


if __name__=='__main__':
    main()
