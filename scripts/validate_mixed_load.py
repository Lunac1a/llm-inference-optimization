"""Preregistered mixed-load experiment; fresh evidence only, no GPU retries."""
import argparse
import asyncio
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import statistics

from stage3_lib import ROOT, OwnedVllm, Telemetry, metrics_snapshot, metric_delta, write_json, gpu_sample, reset_prefix_cache
from validate_hybrid_prefill import config, prompt, launch_environment
from validate_fp8_capacity import hashes
from local_document_qa import port_open

OUT = ROOT / 'artifacts/mixed-load'
PERF = '概括本文档的项目情况。'
WARM = '文档编号是什么？'
QUALITY = '项目负责人是谁，正式预算金额是多少？'
URL = 'http://127.0.0.1:8000'


class Server(OwnedVllm):
    def _paths(self):
        python, _, headers = super()._paths()
        return python, str(ROOT / 'scripts/mixed_vllm_entry.py'), headers


def prepare():
    if OUT.exists():
        raise RuntimeError('Refuse existing evidence')
    import vllm
    import torch
    from transformers import AutoTokenizer
    assert vllm.__version__ == '0.23.0'
    old = hashes(p for p in (ROOT / 'artifacts').rglob('*') if p.is_file())
    frozen = hashes([p for directory in ('configs', 'experiments') for p in (ROOT / directory).rglob('*')
                     if p.is_file() and '__pycache__' not in str(p)] +
                    [p for p in (ROOT / 'scripts').glob('*.py') if 'mixed' not in p.name])
    data = json.loads((ROOT / 'artifacts/hybrid-prefill/materials.json').read_text())
    model = Server(config('hybrid'), OUT / 'unused').resolve_model()
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tokens = {k: [tok.apply_chat_template([{'role': 'user', 'content': prompt(d, q)}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False)
                  for q in (PERF, WARM, QUALITY)] for k, d in data['documents'].items()}
    def common(a, b):
        return sum(1 for _ in itertools.takewhile(lambda x: x[0] == x[1], zip(a, b)))
    cross = {a + ':' + b: common(tokens[a][0], tokens[b][0]) for a, b in itertools.combinations(tokens, 2)}
    same = {k: common(t[0], t[1]) for k, t in tokens.items()}
    assert max(cross.values()) < 16 and min(same.values()) > 16384 * .99
    assert all(len(t) + 128 <= 16640 for ts in tokens.values() for t in ts)
    package = Path(vllm.__file__).parent
    files = ['config/scheduler.py', 'v1/core/sched/scheduler.py', 'engine/arg_utils.py',
             'entrypoints/openai/chat_completion/serving.py', 'entrypoints/openai/chat_completion/protocol.py',
             'v1/attention/backends/triton_attn.py']
    OUT.mkdir()
    write_json(OUT / 'old-evidence-hashes.json', old)
    write_json(OUT / 'frozen-source-hashes.json', frozen)
    write_json(OUT / 'materials.json', data)
    write_json(OUT / 'preflight.json', {'vllm': vllm.__version__, 'torch': torch.__version__, 'model': model,
        'tokens': {k: list(map(len, ts)) for k, ts in tokens.items()}, 'cross': cross, 'same': same,
        'config': config('hybrid'), 'sources': {f: (package / f).read_text() for f in files},
        'package_hashes': {str(package / f): hashlib.sha256((package / f).read_bytes()).hexdigest() for f in files}})
    print('CPU preflight passed', flush=True)


async def request(session, document, question, name, folder, length=128, fixed=True, callback=None):
    body = {'model': 'stage3-document-qa', 'messages': [{'role': 'user', 'content': prompt(document, question)}],
            'temperature': 0, 'seed': 42, 'max_tokens': length, 'ignore_eos': fixed, 'stream': True,
            'return_token_ids': True, 'stream_options': {'include_usage': True, 'continuous_usage_stats': True},
            'chat_template_kwargs': {'enable_thinking': False}}
    row = {'name': name, 'start': time.perf_counter(), 'events': [], 'text': '', 'token_count': 0,
           'finish_reason': None, 'done': False, 'body_sha256': hashlib.sha256(json.dumps(body).encode()).hexdigest()}
    try:
        async with asyncio.timeout(60):
            async with session.post(URL + '/v1/chat/completions', json=body) as response:
                row['status'] = response.status
                if response.status != 200:
                    raise RuntimeError(await response.text())
                async for line in response.content:
                    line = line.decode().strip()
                    if not line.startswith('data:'):
                        continue
                    stamp = time.perf_counter()
                    raw = line[5:].strip()
                    if raw == '[DONE]':
                        row['done'] = True
                        break
                    item = json.loads(raw)
                    row['events'].append({'time': stamp, 'data': item})
                    for choice in item.get('choices', []):
                        content = (choice.get('delta') or {}).get('content') or ''
                        row['text'] += content
                        row['token_count'] += len(choice.get('token_ids') or [])
                        if content and 'first_content' not in row:
                            row['first_content'] = stamp
                        if choice.get('finish_reason'):
                            row['finish_reason'] = choice['finish_reason']
                    if item.get('usage'):
                        row['usage'] = item['usage']
                    if callback:
                        callback(row, stamp)
        assert row['done'] and row['text'] and row['finish_reason'] == ('length' if fixed else 'stop'), row
        assert row['token_count'] == row['usage']['completion_tokens']
        if fixed:
            assert row['token_count'] == length
    except BaseException as exc:
        row['error'] = str(exc)
        raise
    finally:
        row['end'] = time.perf_counter()
        row['e2e'] = row['end'] - row['start']
        row['ttft'] = row.get('first_content', row['end']) - row['start']
        row['output_tok_s'] = row['token_count'] / row['e2e']
        write_json(folder / (name + '.json'), row)
    return row


def content_times(row):
    return [e['time'] for e in row['events'] if any((c.get('delta') or {}).get('content')
            for c in e['data'].get('choices', []))]


def interference(pairs):
    flags = []
    for control, mixed in pairs:
        flags.append({'A': mixed['A_gap'] >= 2 * control['A_gap'] and mixed['A_gap'] - control['A_gap'] >= .250,
                      'B': mixed['B_ttft'] >= 2 * control['B_ttft'] and mixed['B_ttft'] - control['B_ttft'] >= .500})
    return {'pairs': flags, 'triggered': sum(any(f.values()) for f in flags) >= 2}


def labelled_delta(before, after):
    def values(snapshot):
        return {(r['name'], json.dumps(r['labels'], sort_keys=True)): r['value'] for r in snapshot['selected']}
    left, right = values(before), values(after)
    return [{'name': n, 'labels': json.loads(labels), 'delta': right.get((n, labels), 0) - left.get((n, labels), 0)}
            for n, labels in sorted(set(left) | set(right))]


async def trial(session, docs, folder, mixed):
    folder.mkdir()
    reset = reset_prefix_cache(URL)
    write_json(folder / 'reset.json', reset)
    assert reset['ok'], reset
    for key in ('a', 'b'):
        await request(session, docs['doc-' + key], WARM, 'warm-' + key, folder, 16)
    before = metrics_snapshot(URL)
    telemetry = Telemetry(URL, folder / 'telemetry.jsonl')
    telemetry.start()
    triggered = asyncio.Event()
    trigger = {}
    children = []
    def on_a(row, stamp):
        if row['token_count'] >= 16 and not triggered.is_set():
            assert row['token_count'] == 16 and row['finish_reason'] is None
            trigger.update({'time': stamp, 'A_tokens': row['token_count'], 'A_finished': False})
            triggered.set()
    started = time.perf_counter()
    try:
        async with asyncio.timeout(180):
            a_task = asyncio.create_task(request(session, docs['doc-a'], PERF, 'A', folder, callback=on_a))
            children.append(a_task)
            # Propagate A failures instead of waiting forever for its trigger.
            event_task = asyncio.create_task(triggered.wait())
            children.append(event_task)
            await asyncio.wait([a_task, event_task], return_when=asyncio.FIRST_COMPLETED)
            assert triggered.is_set() and not a_task.done(), 'A must still generate at dispatch'
            tasks = [a_task, asyncio.create_task(request(session, docs['doc-b'], PERF, 'B', folder))]
            if mixed:
                tasks.append(asyncio.create_task(request(session, docs['doc-c'], PERF, 'C', folder)))
            children.extend(tasks[1:])
            rows = await asyncio.gather(*tasks)
        by = {r['name']: r for r in rows}
        dispatch = by['C' if mixed else 'B']['start']
        assert by['A']['start'] < dispatch < by['A']['end']
        assert any(e['time'] > dispatch and any(c.get('token_ids') for c in e['data'].get('choices', []))
                   for e in by['A']['events']), 'Missing A progress after insertion'
        if mixed:
            assert by['B']['start'] <= by['C']['start']
        times = content_times(by['A'])
        gaps = [b - a for a, b in zip(times, times[1:]) if b > dispatch]
        assert gaps
        for key in ('A', 'B'):
            assert by[key]['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) >= 16384 * .99
        if mixed:
            assert by['C']['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) == 0
        result = {'mixed': mixed, 'trigger': trigger, 'dispatch': dispatch, 'A_gap': max(gaps), 'A_gaps': gaps,
                  'B_ttft': by['B']['ttft'], 'requests': {k: {x: r.get(x) for x in
                    ('start', 'end', 'ttft', 'e2e', 'output_tok_s', 'token_count', 'usage')} for k, r in by.items()},
                  'throughput': sum(r['token_count'] for r in rows) / (max(r['end'] for r in rows) - started),
                  'metrics_before': before}
    finally:
        for task in children:
            if not task.done():
                task.cancel()
        await asyncio.gather(*children, return_exceptions=True)
        telemetry.stop()
    after = metrics_snapshot(URL)
    result['metrics_after'] = after
    result['metric_deltas'] = labelled_delta(before, after)
    def count(s, name):
        return sum(r['value'] for r in s['selected'] if r['name'] == name)
    for name in ('vllm:num_preemptions_total', 'vllm:request_queue_time_seconds_sum'):
        assert any(r['name'] == name for r in after['selected']), name
    result['preemptions'] = count(after, 'vllm:num_preemptions_total') - count(before, 'vllm:num_preemptions_total')
    write_json(folder / 'summary.json', result)
    assert result['preemptions'] == 0
    return result


async def collect():
    import aiohttp
    if (OUT / 'budget.json').exists():
        raise RuntimeError('Refuse GPU rerun')
    assert not port_open(config('hybrid')), 'Port occupied'
    docs = json.loads((OUT / 'materials.json').read_text())['documents']
    started = time.time()
    write_json(OUT / 'budget.json', {'started': started, 'deadline': started + 1800})
    write_json(OUT / 'protocol.json', {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'sources': hashes([ROOT / 'docs/mixed-load-plan.md', Path(__file__), ROOT / 'scripts/mixed_vllm_entry.py'])})
    def ensure():
        assert time.time() + 300 < started + 1800, 'Budget exhausted'
    all_results = {}
    try:
        async with aiohttp.ClientSession() as session:
            for mode in ('full', 'chunk'):
                ensure()
                folder = OUT / mode
                folder.mkdir()
                server = Server(config('hybrid'), folder / 'server')
                try:
                    os.environ['MIXED_MODE'] = mode
                    print('START ' + mode, flush=True)
                    with launch_environment('hybrid', folder):
                        server.start(timeout=180)
                    for i in range(2):
                        await request(session, {'text': '短文。天气晴朗。'}, '写一句话。', 'short-' + str(i), folder, 16)
                    pairs = []
                    for rep in range(3):
                        pair = {}
                        for mixed in ([False, True] if rep != 1 else [True, False]):
                            ensure()
                            label = f'r{rep + 1}-' + ('mixed' if mixed else 'control')
                            pair[mixed] = await trial(session, docs, folder / label, mixed)
                            print(json.dumps({'mode': mode, 'trial': label, 'A_gap': pair[mixed]['A_gap'],
                                              'B_ttft': pair[mixed]['B_ttft']}), flush=True)
                        pairs.append((pair[False], pair[True]))
                    decision = interference(pairs)
                    all_results[mode] = {'pairs': pairs, 'decision': decision}
                    write_json(OUT / 'results.json', all_results)
                    ensure()
                    assert reset_prefix_cache(URL)['ok']
                    for key, doc in docs.items():
                        await request(session, doc, QUALITY, 'quality-' + key, folder, fixed=False)
                    log = (folder / 'server/server.log').read_text(errors='replace')
                    assert 'HYBRID_SCHEDULE' in log and 'HYBRID_FORWARD cached_fp8' in log
                    if mode == 'full':
                        assert 'HYBRID_FORWARD cold_long_bf16' in log
                finally:
                    server.stop()
                    os.environ.pop('MIXED_MODE', None)
                assert not port_open(config('hybrid'))
                if mode == 'full' and not decision['triggered']:
                    print('NO INTERFERENCE GATE: STOP', flush=True)
                    break
    except BaseException as exc:
        write_json(OUT / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        write_json(OUT / 'cleanup.json', {'elapsed_seconds': time.time() - started,
            'port_closed': not port_open(config('hybrid')), 'gpu': gpu_sample(),
            'compute_processes': subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv'],
                                               capture_output=True, text=True).stdout})


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'run'])
    phase = parser.parse_args().phase
    prepare() if phase == 'prepare' else asyncio.run(collect())
