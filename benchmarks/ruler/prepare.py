"""Download official RULER sources' data and invoke unchanged v1 generators."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
from common import ROOT, UPSTREAM, MODEL, LENGTHS, digest, write_json, append

sys.path.insert(0, str(ROOT / 'deps'))

def download(url, path):
    if path.exists():
        return
    request = urllib.request.Request(url, headers={'User-Agent': 'RULER-local-evaluation/1.0'})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read()
    path.write_bytes(data)

def corpus():
    import html2text
    from bs4 import BeautifulSoup
    import nltk
    location = UPSTREAM / 'scripts/data/synthetic/json'
    raw = ROOT / 'corpus-downloads'
    raw.mkdir(exist_ok=True)
    urls = (location / 'PaulGrahamEssays_URLs.txt').read_text().splitlines()
    def fetch(item):
        index, source = item
        # Preserve the official URL list; use HTTPS and raw.githubusercontent for transport.
        url = source.replace('http://www.paulgraham.com/', 'https://www.paulgraham.com/')
        url = url.replace('https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/',
                          'https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/main/')
        path = raw / f'{index:03d}.raw'
        download(url, path)
        if '.html' in source:
            h = html2text.HTML2Text()
            h.ignore_images = h.ignore_tables = h.escape_all = True
            h.reference_links = h.mark_code = False
            soup = BeautifulSoup(path.read_bytes().decode('unicode_escape', 'utf-8'), 'html.parser')
            body = soup.find('font')
            assert body is not None, source
            text = h.handle(str(body))
            group = 'html'
        else:
            text = path.read_text(encoding='utf-8')
            group = 'repo'
        return {'source': source, 'transport_url': url, 'sha256': digest(path),
                'group': group, 'name': source.rsplit('/', 1)[-1].replace('.html', '.txt'), 'text': text}
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(fetch, enumerate(urls)))
    # Official downloader concatenates sorted repo text first, then sorted HTML text.
    text = ''.join(r['text'] for r in sorted(rows, key=lambda r: (0 if r['group']=='repo' else 1, r['name'])))
    write_json(location / 'PaulGrahamEssays.json', {'text': text})
    squad = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'
    hotpot = 'https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json'
    for url, name in [(squad, 'squad.json'), (hotpot, 'hotpotqa.json')]:
        download(url, location / name)
        json.loads((location / name).read_text())
    os.environ['NLTK_DATA'] = str(ROOT / 'nltk_data')
    nltk.data.path.insert(0, os.environ['NLTK_DATA'])
    for name in ['punkt', 'punkt_tab']:
        assert nltk.download(name, download_dir=os.environ['NLTK_DATA'], quiet=True)
    write_json(ROOT / 'corpus-manifest.json', {'essays': [{k:v for k,v in r.items() if k!='text'} for r in rows],
        'qa_sources': {'squad': squad, 'hotpotqa': hotpot},
        'files': {p.name: digest(p) for p in location.glob('*.json')}})
    print('Corpus prepared:', len(rows), 'essays', flush=True)

def generate():
    import yaml
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    spec = importlib.util.spec_from_file_location('ruler_data_constants', UPSTREAM / 'scripts/data/synthetic/constants.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    tasks = yaml.safe_load((UPSTREAM / 'scripts/synthetic.yaml').read_text())
    data = ROOT / 'data'; data.mkdir(exist_ok=True)
    env = {**os.environ, 'PYTHONPATH': str(ROOT/'deps'), 'NLTK_DATA': str(ROOT/'nltk_data'), 'TOKENIZERS_PARALLELISM':'false'}
    for length in LENGTHS:
        for task, cfg in tasks.items():
            definition = module.TASKS[cfg['task']]
            template = tokenizer.apply_chat_template([{'role':'user', 'content':definition['template']}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False) + definition.get('answer_prefix', '')
            folder = data / str(length)
            target = folder / task / 'validation.jsonl'
            if not target.exists():
                args = [sys.executable, str(UPSTREAM/f"scripts/data/synthetic/{cfg['task']}.py"),
                    '--save_dir', str(folder), '--save_name', task, '--subset','validation',
                    '--tokenizer_path', MODEL, '--tokenizer_type', 'hf', '--max_seq_length', str(length),
                    '--tokens_to_generate',str(definition['tokens_to_generate']), '--num_samples','11',
                    '--random_seed','42', '--template',template]
                for key,value in cfg['args'].items(): args += ['--'+key,str(value)]
                log = ROOT / f'prepare-{length}-{task}.log'
                with log.open('w') as stream:
                    subprocess.run(args, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=300)
                append(ROOT/'preparation-commands.jsonl', {'task':task,'length':length,'argv':args})
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            assert len(rows)==11
            for index,row in enumerate(rows):
                prompt = row['input'] + row.get('answer_prefix','')
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                allowance = definition['tokens_to_generate']
                assert len(ids)+allowance<=16640, (task,length,len(ids),allowance)
                assert row['outputs'] and all(isinstance(x,str) and x for x in row['outputs'])
                row.update(id=f'{length}:{task}:{index}', sample=index, task=task, category=cfg['task'],
                           target_length=length, prompt_ids=ids, max_tokens=allowance, prompt_tokens=len(ids))
            output = folder / f'{task}.jsonl'
            output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
            print('Prepared',length,task,len(rows), flush=True)
    files = {str(p.relative_to(ROOT)):digest(p) for p in data.glob('*/*.jsonl')}
    write_json(ROOT/'data-manifest.json', {'upstream_commit':subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip(),
        'model':MODEL,'samples_per_task_length':11,'formal_candidates':list(range(10)), 'pilot_sample':10,
        'lengths':LENGTHS,'tasks':list(tasks),'files':files,'template':'Qwen3 nonthinking chat template + official assistant answer prefix; completions prompt IDs',
        'sources':{str(p.relative_to(UPSTREAM)):digest(p) for p in (UPSTREAM/'scripts').rglob('*.py')}})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['corpus','generate']); args=parser.parse_args()
    {'corpus':corpus,'generate':generate}[args.phase]()
