import importlib.util
import subprocess
from pathlib import Path

import pytest

from anyrsi import Patch
from anyrsi_guard import PolicyGuard
from anyrsi_llm.line_hunks import line_hunks
from anyrsi_runtime.group import merge_patches


def benchmark_module(name):
    p = Path(__file__).resolve().parents[1] / 'benchmarks/self_propose_llm' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_splits_by_content_not_id(seed):
    P = benchmark_module('controlled_pool').ExpressionPool
    pool = P('abcdefgh', seed)
    sets = [set(pool.rows(s)) for s in ['train', 'holdout', 'final']]
    assert list(map(len, sets)) == [440, 24, 48]
    assert sum(map(len, sets)) == len(set.union(*sets))
    assert P('abcdefgh', seed).digest == pool.digest


def test_request_cap_counts_proposals_and_transport_retries(tmp_path, monkeypatch):
    import io
    import urllib.error

    import anyrsi_llm as llm
    meter = benchmark_module('request_budget').RequestBudget(2, tmp_path/'events.jsonl')
    attempts = []
    def send(req, timeout):
        attempts.append(req)
        if len(attempts) == 1:
            raise urllib.error.URLError("temporary")
        return io.BytesIO(b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],'
                          b'"usage":{"total_tokens":9}}')
    monkeypatch.setattr(llm.urllib.request, 'urlopen', send)
    monkeypatch.setattr(llm.time, 'sleep', lambda *_: None)
    mark = llm.request_observer.set(meter)
    try:
        with meter.phase('solver'):
            assert llm.chat('https://example.invalid', 'model', [])['text'] == 'ok'
        with pytest.raises(RuntimeError, match='budget'):
            llm.chat('https://example.invalid', 'model', [])
        with meter.phase('final'):
            llm.chat('https://example.invalid', 'model', [])
    finally:
        llm.request_observer.reset(mark)
    assert len(attempts) == 3 and meter.report()['by_role'] == {'solver': 2, 'final': 1}
    assert meter.report()['reported_tokens']['total_tokens'] == 18


def test_two_edits_in_one_file_can_be_compared_and_merged(tmp_path):
    original = ''.join(f'line {i}\n' for i in range(30))
    updated = original.replace('line 2\n', 'alpha\n').replace('line 22\n', 'beta\n')
    ops = line_hunks('file.txt', original, updated, 4)
    assert len(ops) == 2
    patches = [Patch(target='harness', ops=[o]) for o in ops]
    assert merge_patches(patches + patches).ops == ops
    f = tmp_path/'file.txt'
    for op in ops:
        f.write_text(original)
        subprocess.run(['git','apply','-',], cwd=tmp_path, input=op['unified'], text=True, check=True)
        assert ('alpha' in f.read_text()) != ('beta' in f.read_text())
    f.write_text(original)
    for op in ops:
        subprocess.run(['git','apply','-'], cwd=tmp_path, input=op['unified'], text=True, check=True)
    assert f.read_text() == updated
    with pytest.raises(ValueError, match='limit'):
        line_hunks('file.txt', original, 'rewrite\n', 4)


def test_hunk_cannot_bypass_python_ast_guard():
    ops = line_hunks('agent.py', 'run()\n', 'import os\nrun()\n', 3)
    assert not PolicyGuard(check_harness_code=True).inspect(Patch(target='harness', ops=ops)).allowed


def test_a_real_runner_at_the_http_boundary(tmp_path, monkeypatch):
    import asyncio
    import io
    import json
    import anyrsi_llm

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'benchmarks/self_propose_llm'))
    from loop_llm import run_arm

    monkeypatch.setenv('TEST_KEY', 'mock-key')
    def fake_http(req, timeout):
        result = {'choices': [{'message': {'content': '1 + 1 = 2'}, 'finish_reason': 'stop'}],
                  'usage': {'total_tokens': 9}}
        return io.BytesIO(json.dumps(result).encode())
    monkeypatch.setattr(anyrsi_llm.urllib.request, 'urlopen', fake_http)
    for arm in ['bon:patch', 'atom:patch', 'grad:patch']:
        out = tmp_path/arm
        result = asyncio.run(run_arm(arm, out, 0, 'mock', 'TEST_KEY', 100, 2, True, 0, 42, 2))
        assert result['budget']['search_used'] == 42 and result['budget']['budget_exhausted']
        assert result['budget']['by_role']['final'] == 48
        assert result['budget']['reported_tokens']['total_tokens'] == 90 * 9
        assert len((out/'sealed-cases.jsonl').read_text().splitlines()) == 48

def test_shared_panel_compares_whole_best_atom_and_merge(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'benchmarks/terminal_directions'))
    from policy import pick
    parent = {"x": 0., "y": 0., "z": 1.}
    first = {"x": 1., "y": 0., "z": 1.}
    second = {"x": 0., "y": 1., "z": 1.}
    assert pick(parent, [parent, first], [first, second]) == {"whole": 1, "best_atom": 0, "merge": [0, 1]}
    assert pick(parent, [parent, parent], [parent, parent]) == {"whole": None, "best_atom": None, "merge": []}
    with pytest.raises(ValueError):
        pick(parent, [parent, first], [first, {"x": 0.}])

def test_template_forwards_only_an_environment_reference():
    import yaml

    path = Path(__file__).resolve().parents[1]/'examples/rsihub_weights/anyrsi.step1.yaml'
    agent = yaml.safe_load(path.read_text())['components']['agent']['with']
    assert 'ANYRSI_HARBOR_API_KEY' in agent['env']
    argv = [v.format(repo='/tmp/repo') for v in agent['extra_args']]
    assert 'OPENAI_API_KEY=${ANYRSI_HARBOR_API_KEY}' in argv
