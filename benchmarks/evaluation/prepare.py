"""Generate new inputs with unchanged official generators; never overwrite data."""
import importlib.util
import os
import subprocess
import sys
from config import *


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    assert subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip() == UPSTREAM_COMMIT
    # Reuse already verified public corpora; do not touch historical data or tools.
    old = read(PREVIOUS / 'data-manifest.json')
    for name, sha in old['sources'].items(): assert digest(UPSTREAM / name) == sha
    corpus = read(PREVIOUS / 'corpus-manifest.json')
    for name, sha in corpus['files'].items(): assert digest(UPSTREAM / 'scripts/data/synthetic/json' / name) == sha
    sys.path.insert(0, str(DEPS))
    import yaml
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    spec = importlib.util.spec_from_file_location('official_data', UPSTREAM / 'scripts/data/synthetic/constants.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    tasks = yaml.safe_load((UPSTREAM / 'scripts/synthetic.yaml').read_text())
    old_rows = [r for p in (PREVIOUS / 'data').glob('*/*.jsonl') for r in rows(p)]
    old_hashes = {prompt_hash(r) for r in old_rows}
    old_questions = {r['input'].rsplit('Question:', 1)[-1].strip() for r in old_rows if r['category'] == 'qa'}
    seen = set(); qa_questions = set(); manifest = {}
    env = {**os.environ, 'PYTHONPATH': str(DEPS), 'NLTK_DATA': str(NLTK), 'TOKENIZERS_PARALLELISM': 'false'}
    for length in LENGTHS:
        for task, cfg in tasks.items():
            definition = module.TASKS[cfg['task']]
            template = tokenizer.apply_chat_template([{'role': 'user', 'content': definition['template']}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False) + definition.get('answer_prefix', '')
            folder = ROOT / 'data' / str(length)
            args = [sys.executable, str(UPSTREAM / f"scripts/data/synthetic/{cfg['task']}.py"),
                '--save_dir', str(folder), '--save_name', task, '--subset', 'validation',
                '--tokenizer_path', MODEL, '--tokenizer_type', 'hf', '--max_seq_length', str(length),
                '--tokens_to_generate', str(definition['tokens_to_generate']), '--num_samples', str(N),
                '--random_seed', '20260919', '--template', template]
            # QA chooses question by ordinal; use an unobserved ordinal range, not just a new seed.
            if cfg['task'] == 'qa': args += ['--pre_samples', '100']
            for key, value in cfg['args'].items(): args += ['--' + key, str(value)]
            append(ROOT / 'preparation-commands.jsonl', {'task': task, 'length': length, 'argv': args})
            with (ROOT / f'prepare-{length}-{task}.log').open('x') as stream:
                subprocess.run(args, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
            data = rows(folder / task / 'validation.jsonl'); assert len(data) == N
            for sample, row in enumerate(data):
                ids = tokenizer.encode(row['input'] + row.get('answer_prefix', ''), add_special_tokens=False)
                assert len(ids) + definition['tokens_to_generate'] <= 16640
                assert row['outputs'] and all(isinstance(x, str) and x for x in row['outputs'])
                row.update(id=f'{length}:{task}:{sample}', task=task, sample=sample,
                    category=cfg['task'], target_length=length, prompt_ids=ids,
                    max_tokens=definition['tokens_to_generate'], prompt_tokens=len(ids))
                sha = prompt_hash(row)
                assert sha not in old_hashes and sha not in seen, 'Repeated prompt; stop without substitution'
                seen.add(sha)
                if cfg['task'] == 'qa':
                    question = row['input'].rsplit('Question:', 1)[-1].strip()
                    assert question not in old_questions, 'Previously observed QA question'
                    qa_questions.add(question)
            target = folder / f'{task}.jsonl'
            target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data), encoding='utf-8')
            manifest[str(target)] = digest(target)
            print(f'Prepared {length} {task}: {N}', flush=True)
    # Timing uses an old, fixed retrieval input, disjoint from the new quality suite.
    timing = {str(length): rows(PREVIOUS / f'data/{length}/niah_single_1.jsonl')[0] for length in LENGTHS}
    write(ROOT / 'timing-inputs.json', timing)
    write(ROOT / 'data-manifest.json', {'files': manifest, 'n': N, 'tasks': list(tasks), 'lengths': LENGTHS,
        'inputs': len(seen), 'fresh_prompt_hashes': True, 'qa_questions_disjoint': True,
        'unique_qa_questions': len(qa_questions), 'generator_seed': 20260919, 'qa_pre_samples': 100,
        'upstream_commit': UPSTREAM_COMMIT, 'model': MODEL,
        'upstream_files': {str(UPSTREAM / name): sha for name, sha in old['sources'].items()},
        'corpus_files': {str(UPSTREAM / 'scripts/data/synthetic/json' / name): sha for name, sha in corpus['files'].items()}})


if __name__ == '__main__': main()
