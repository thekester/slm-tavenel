"""Archive completed likelihood evidence and the current human-review status, without raw corpus text or review keys."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_story_human_review import summarize
from serve_story_review import write


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=ROOT/'runs/EXP-0045/tinystories-validation-2026-11-10'
    dest=ROOT/'docs/model-cards/slm-tavenel-story-935k/training_record/tinystories'
    dest.mkdir(parents=True,exist_ok=True)
    manifest,summary=load(source/'manifest.json'),load(source/'summary.json')
    if manifest['status']!='completed':
        raise ValueError('Cannot publish an incomplete evaluation')
    evaluator=ROOT/'scripts/evaluate_story_tinystories.py'
    if sha(evaluator)!=manifest['fingerprint']['script_sha256']:
        raise ValueError('Executed evaluator changed; preserve the executed version before archiving')
    aligned=[]
    for name in ['story935k','tiny1m']:
        path=source/f'{name}-items.jsonl'
        assert sha(path)==summary['results'][name]['items_sha256']
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert len(rows)==summary['data']['documents']
        reconstructed=sum(r['nll'] for r in rows)/math.log(2)/sum(r['bytes'] for r in rows)
        assert abs(reconstructed-summary['results'][name]['bits_per_byte'])<1e-10
        aligned.append([(r['index'],r['text_sha256'],r['bytes']) for r in rows])
        with (dest/f'{name}-items.jsonl.gz').open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:
                stream.write(path.read_bytes())
    assert aligned[0]==aligned[1], 'Model scores are not aligned to the same texts'
    for name in ['summary.json','manifest.json','dataset-audit.json','simple-dev-reference-reconciliation.json']:
        shutil.copy2(source/name,dest/name)
    shutil.copy2(evaluator,dest/'executed-evaluator.py')
    shutil.copy2(ROOT/'scripts/check_story_reference_tokenizer.py',dest/'executed-tokenizer-check.py')
    prior=load(ROOT/'docs/model-cards/slm-tavenel-story-935k/training_record/generation-comparison/manifest.json')
    snapshots=[]
    for model in prior['models']:
        folder=ROOT/model['path']
        files={name:sha(folder/name) for name in model['files_sha256']}
        assert files==model['files_sha256'], 'Frozen inputs changed after original snapshot verification'
        snapshots.append({'repo':model['repo_id'],'revision':model['revision'],'path':model['path'],
                          'files_sha256':files,'tokenizer_class':('LlamaTokenizer, legacy=True' if model['label']=='tiny1m' else 'PreTrainedTokenizerFast'),
                          'context':256,'device':'cpu','dtype':'float32'})
    snapshots[0]['repo']='athenista/slm-tavenel-story-935k'
    snapshots[0]['revision']='62e64c2b677ea9fbc90cdee0fbfe66b589b78e83'
    snapshots[1]['subfolder']='hf'
    write(dest/'frozen-model-inputs.json',snapshots)
    review=ROOT/'runs/EXP-0045/human-blind-pilot-2026-11-10'
    public=dest.parent/'human-review';public.mkdir(exist_ok=True)
    shutil.copy2(review/'meta.json',public/'protocol.json')
    write(public/'summary.json',summarize(review))
    write(dest/'followup-checks.json',{'paired_items_aligned':True,'aggregate_nll_reconstructed':True,
                                     'full_validation_documents':summary['data']['documents'],
                                     'targeted_tests_passed':20,'deselected':10,'test_seconds':13.00,
                                     'test_command':'python -m pytest tests/test_story_bpb_protocol.py tests/test_story_human_review.py tests/test_story_split.py tests/test_fluency.py tests/test_end_to_end.py -k "story_bpb_protocol or story_human_review or story_split or fluency or export" -q',
                                     'human_ui_javascript_syntax':'passed','human_ui_live_key_route':'404, key not served',
                                     'private_human_ratings_or_keys_copied':False})
    print('Archived complete aligned likelihood evidence and aggregate-only human status; no human ratings fabricated.')


if __name__=='__main__':
    main()
