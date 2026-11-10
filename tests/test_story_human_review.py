"""Protect the blind boundary and prevent absent ratings from becoming apparent human evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
from http.server import HTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

SCRIPTS = Path(__file__).parents[1]/"scripts"
sys.path.insert(0, str(SCRIPTS))
import serve_story_review as review
from analyze_story_human_review import summarize


def bundle(tmp_path):
    source = tmp_path/'source.jsonl'
    source.write_text('\n'.join(json.dumps({'id':'prefix1','temperature':0,'seed':0,'model':model,
                                         'prompt':'A fox found a box.','completion':' It was a small box.',
                                         'stop_reason':'byte_budget'})
                                for model in ['hidden-model-one','hidden-model-two'])+'\n')
    directory = tmp_path/'bundle'
    review.prepare(directory, source)
    return directory


def test_review_server_never_serves_model_key(tmp_path):
    directory=bundle(tmp_path)
    server=HTTPServer(('127.0.0.1',0),review.handler(directory,'test-token'))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        base=f'http://127.0.0.1:{server.server_port}'
        with urlopen(base+'/api/cases') as response:
            body=response.read().decode()
        assert 'hidden-model' not in body
        with pytest.raises(HTTPError) as err:
            urlopen(base+'/key.json')
        assert err.value.code==404
    finally:
        server.shutdown();server.server_close();thread.join()


def test_no_ratings_never_produce_a_human_preference_score(tmp_path):
    report=summarize(bundle(tmp_path))
    assert report['status']=='awaiting_human_ratings'
    assert report['complete_blind_human_forms']==0
    assert report['preference'] is None and report['trait_means'] is None


def test_partial_or_machine_forms_do_not_count_as_completed_human_review(tmp_path):
    directory=bundle(tmp_path)
    (directory/'ratings-test.json').write_text(json.dumps({'finished':True,'reviewer_type':'machine','ratings':{}}))
    assert summarize(directory)['complete_blind_human_forms']==0
    with pytest.raises(ValueError):
        review.validate_rating({'rater':'reader','case_id':'one','human':True,'blind':True,
                                'preference':'A','ratings':{'A':{'grammar':True}}},{'one'})
