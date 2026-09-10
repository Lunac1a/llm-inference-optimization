"""Fixed eager versus decode-only graph trial using existing streaming collector."""
import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from local_document_qa import configuration, port_open
from stage3_lib import ROOT, OwnedVllm, write_json, gpu_sample
from validate_fp8_capacity import hashes
from validate_kv_offload import memory, Monitor, snapshot
from validate_mixed_load import request, labelled_delta, PERF, QUALITY, prompt

OUT = ROOT / 'artifacts/cuda-graph'


class Server(OwnedVllm):
    def _paths(self):
        python, _, headers = super()._paths()
        return python, str(ROOT / 'scripts/cuda_graph_entry.py'), headers


@contextmanager
def launch(mode, folder):
    values = {'CUDA_GRAPH_MODE': mode, 'CUDA_GRAPH_LAUNCH': str(folder / 'actual-launch.json'),
              'VLLM_PLUGINS': ''}
    before = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try: yield
    finally:
        for k, v in before.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def prepare():
    import vllm
    from transformers import AutoTokenizer
    assert vllm.__version__ == '0.23.0'
    if OUT.exists(): raise RuntimeError('Refuse existing evidence')
    assert memory()['MemAvailable'] >= 8 * 2**30
    data = json.loads((ROOT / 'artifacts/hybrid-prefill/materials.json').read_text())
    model = Server(configuration(), OUT / 'unused').resolve_model()
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    text = tok.decode(tok.encode(data['documents']['doc-a']['text'], add_special_tokens=False)[:2048])
    assert len(tok.encode(text, add_special_tokens=False)) == 2048
    doc = {'text': text}
    count = len(tok.apply_chat_template([{'role': 'user', 'content': prompt(doc, PERF)}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False))
    old = hashes(p for p in (ROOT / 'artifacts').rglob('*') if p.is_file())
    frozen = hashes([p for d in ('configs', 'experiments', 'scripts') for p in (ROOT / d).rglob('*')
        if p.is_file() and '__pycache__' not in str(p) and 'cuda_graph' not in p.name])
    package = Path(vllm.__file__).parent
    names = ['config/compilation.py', 'config/observability.py', 'compilation/cuda_graph.py',
             'v1/worker/gpu_model_runner.py', 'v1/cudagraph_dispatcher.py', 'engine/arg_utils.py']
    sources = {str(package / n): hashlib.sha256((package / n).read_bytes()).hexdigest() for n in names}
    OUT.mkdir()
    write_json(OUT / 'old-evidence-hashes.json', old)
    write_json(OUT / 'frozen-source-hashes.json', frozen)
    write_json(OUT / 'preflight.json', {'model': model, 'config': configuration(), 'memory': memory(),
        'vllm': vllm.__version__, 'prompt_tokens': count, 'package_hashes': sources,
        'sources': {n: (package / n).read_text() for n in names}})
    write_json(OUT / 'materials.json', {'performance': doc, 'quality': {k: {'text':
        f"项目负责人：{d['facts']['owner']}。正式预算：{d['facts']['budget']}。", 'facts': d['facts']}
        for k, d in data['documents'].items()}})
    print('CPU preflight passed', count, flush=True)


async def run():
    import aiohttp
    if (OUT / 'budget.json').exists(): raise RuntimeError('Refuse rerun')
    assert not port_open(configuration())
    pre = json.loads((OUT / 'preflight.json').read_text())
    materials = json.loads((OUT / 'materials.json').read_text())
    started = time.time()
    write_json(OUT / 'budget.json', {'start': started, 'deadline': started + 1200})
    write_json(OUT / 'protocol.json', {'git_head': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'hashes': hashes([Path(__file__), ROOT / 'scripts/cuda_graph_entry.py', ROOT / 'docs/cuda-graph-plan.md'])})
    def ensure(reserve=120):
        assert time.time() + reserve < started + 1200, 'Budget'
        m = memory()
        assert m['MemAvailable'] >= 2**30 and m['swap_used'] - pre['memory']['swap_used'] <= 256 * 2**20, m
    try:
        async with aiohttp.ClientSession() as session:
            for rep, order in enumerate((('graph','eager'), ('eager','graph'), ('graph','eager')), 1):
                for mode in order:
                    ensure(300)
                    folder = OUT / f'r{rep}-{mode}'
                    folder.mkdir()
                    server, monitor = Server(configuration(), folder / 'server'), None
                    try:
                        print('START ' + folder.name, flush=True)
                        with launch(mode, folder): server.start(timeout=180)
                        monitor = Monitor(folder, server.process.pid)
                        monitor.start()
                        for n in range(2):
                            await request(session, materials['performance'], PERF, f'warm-{n}', folder, 32)
                        waves = []
                        for concurrency, repeats in ((1, 3), (4, 1)):
                            for repeat in range(repeats):
                                ensure()
                                before = snapshot()
                                requests = await asyncio.gather(*[request(session, materials['performance'], PERF,
                                    f'c{concurrency}-{repeat}-{i}', folder, 512) for i in range(concurrency)])
                                after = snapshot()
                                delta = labelled_delta(before, after)
                                preempt = sum(r['delta'] for r in delta if r['name'] == 'vllm:num_preemptions_total')
                                passed = preempt == 0 and all(r['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
                                    >= .9 * r['usage']['prompt_tokens'] for r in requests)
                                waves.append({'concurrency': concurrency, 'repeat': repeat, 'pass': passed,
                                    'names': [r['name'] for r in requests], 'before': before, 'after': after, 'delta': delta})
                                write_json(folder / 'waves.json', waves)
                                print(json.dumps({'run': folder.name, 'concurrency': concurrency, 'repeat': repeat,
                                    'e2e': [r['e2e'] for r in requests], 'pass': passed}), flush=True)
                                assert passed, 'Cache/preemption control failed'
                        if rep == 1:
                            for key, doc in materials['quality'].items():
                                ensure()
                                await request(session, doc, QUALITY, 'quality-' + key, folder, fixed=False)
                        await asyncio.sleep(2)
                        log = (folder / 'server/server.log').read_text(errors='replace')
                        runtime = 'FULL' if mode == 'graph' else 'NONE'
                        matched = re.findall(r'\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*' + runtime + r'\s*\|\s*\d+\s*\|', log)
                        write_json(folder / 'runtime-evidence.json', {'mode': runtime, 'rows': matched})
                        assert matched, 'Missing native runtime stats'
                    finally:
                        if monitor: monitor.stop()
                        server.stop()
                    assert not port_open(configuration())
    except BaseException as exc:
        write_json(OUT / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        write_json(OUT / 'cleanup.json', {'elapsed': time.time()-started, 'port_closed': not port_open(configuration()),
            'gpu': gpu_sample(), 'memory': memory(), 'compute_processes': subprocess.run(['nvidia-smi',
                '--query-compute-apps=pid,process_name', '--format=csv'], capture_output=True, text=True).stdout})


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    {'prepare': prepare, 'run': lambda: asyncio.run(run())}[sys.argv[1]]()
