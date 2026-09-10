"""New preregistered partial-eviction comparison using the API profile."""
import asyncio
import json
from pathlib import Path
import signal
import subprocess
import sys
import urllib.request
import run_kv_offload_recovery as recovery
from cpu_kv_api import CpuKvServer
from stage3_lib import ROOT, write_json
from validate_fp8_capacity import hashes

original = recovery.original
OUT = ROOT / 'artifacts/kv-offload-integration'
recovery.OUT = original.OUT = OUT
original.__file__ = __file__
old_controls = original.controls


def controls(delta, mode, phase, prompt_tokens):
    result = old_controls(delta, mode, phase, prompt_tokens)
    if phase == 'revisit':
        partial = 0 <= result['local_hit_tokens'] < .9 * prompt_tokens
        if mode == 'baseline':
            result['pass'] = partial and result['external_hit_tokens'] == 0 and result['restored_bytes'] == 0 and result['preemptions'] == 0
        else:
            result['pass'] = result['pass'] and partial
    return result


def check_pair(folder):
    peer = folder.with_name(folder.name.split('-')[0] + ('-baseline' if folder.name.endswith('offload') else '-offload'))
    if not (peer / 'measurement.json').exists():
        return
    a, b = [json.loads((p / 'measurement.json').read_text()) for p in (folder, peer)]
    matches = []
    for x, y in zip(a['requests'], b['requests']):
        if x['phase'] == 'revisit':
            assert x['document'] == y['document'] and y['phase'] == 'revisit'
            diff = abs(x['controls']['local_hit_tokens'] - y['controls']['local_hit_tokens'])
            matches.append({'document': x['document'], 'local_hit_difference': diff, 'pass': diff <= 16})
    write_json(OUT / (folder.name.split('-')[0] + '-pair-control.json'), matches)
    assert len(matches) == 3 and all(r['pass'] for r in matches), matches


class Server(CpuKvServer, recovery.Server):
    def start(self, timeout=180):
        if self.run_dir.parent.name.endswith('offload'):
            # Direct method reuse ensures the delivered profile owns the real launch.
            result = CpuKvServer.start(self, timeout)
            if self.run_dir.parent.name in ('r1-offload', 'r3-offload'):
                payload = {'model': 'stage3-document-qa', 'messages': [{'role': 'user', 'content': '只回答：服务正常'}],
                           'max_tokens': 32, 'temperature': 0, 'stream': False,
                           'chat_template_kwargs': {'enable_thinking': False}}
                req = urllib.request.Request(self.base_url + '/v1/chat/completions',
                    json.dumps(payload).encode(), {'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    answer = json.load(response)
                write_json(self.run_dir.parent / 'api-nonstream.json', answer)
                assert answer['choices'][0]['finish_reason'] == 'stop' and '服务正常' in answer['choices'][0]['message']['content']
            return result
        return recovery.Server.start(self, timeout)

    def stop(self):
        result = super().stop()
        path = self.run_dir.parent / 'measurement.json'
        if path.exists() and len(json.loads(path.read_text())['requests']) == 6:
            check_pair(self.run_dir.parent)
        return result


original.Server = Server
original.controls = controls


def run():
    if (OUT / 'execution.json').exists():
        raise RuntimeError('Refuse rerun')
    write_json(OUT / 'integration-sources.json', hashes([Path(__file__), ROOT / 'scripts/cpu_kv_api.py',
        ROOT / 'scripts/local_document_qa.py', ROOT / 'docs/kv-offload-integration-plan.md']))
    write_json(OUT / 'execution.json', {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'transport': 'scoped upstream single-copy adapter', 'latency_gates': 'unchanged',
        'cache_controls': 'partial eviction plus paired GPU retained-prefix matching',
        'plan': 'docs/kv-offload-integration-plan.md'})
    asyncio.run(original.run())


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    {'prepare': original.prepare, 'copy-probe': recovery.probe, 'run': run}[sys.argv[1]]()
