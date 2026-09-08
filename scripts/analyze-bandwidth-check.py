"""Summarize all retained Nsight Compute samples; never run GPU work."""
import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/stage2-bandwidth'
DRAM = 'dram__throughput.avg.pct_of_peak_sustained_elapsed'
SM = 'sm__throughput.avg.pct_of_peak_sustained_elapsed'
TIME = 'gpu__time_duration.sum'
READ = 'dram__bytes_read.sum'
WRITE = 'dram__bytes_write.sum'


def parse(path):
    text = path.read_text()
    start = text.find('"ID","Process ID"')
    if start < 0:
        return []
    rows = csv.DictReader(io.StringIO(text[start:]))
    groups = {}
    for row in rows:
        if not row.get('Metric Name'):
            continue
        expected_units = {DRAM: '%', SM: '%', TIME: 'ns', READ: 'byte', WRITE: 'byte'}
        assert row['Metric Unit'] == expected_units[row['Metric Name']]
        key = (row['Process ID'], row['ID'])
        item = groups.setdefault(key, {'id': row['ID'], 'kernel': row['Kernel Name'],
                                      'grid': row['Grid Size'], 'block': row['Block Size'], 'metrics': {}})
        item['metrics'][row['Metric Name']] = float(row['Metric Value'].replace(',', ''))
    for item in groups.values():
        m = item['metrics']
        assert set(m) == {DRAM, SM, TIME, READ, WRITE}, m
        assert m[TIME] > 0
        item['read_GB_per_s'] = m[READ] / m[TIME]
        item['total_GB_per_s'] = (m[READ] + m[WRITE]) / m[TIME]
        item['strong_indicator'] = m[DRAM] >= 80 and m[SM] <= 60
    return list(groups.values())


def main():
    captures = []
    for folder in sorted(p for p in OUT.glob('capture-*') if p.is_dir()):
        kernels = parse(folder / 'counters-installed-export.csv')
        requests = []
        for path in sorted(folder.rglob('requests.jsonl')):
            requests.extend(json.loads(s) for s in path.read_text().splitlines())
        item = {'capture': folder.name, 'kernels': kernels, 'request_count': len(requests),
                'successful_requests': sum(r['success'] and not r['error'] for r in requests),
                'fixed_lengths_valid': all(r['prompt_len'] == 512 and r['output_tokens'] == 128 for r in requests)}
        if (folder / 'end.json').exists():
            item['wall_seconds'] = json.loads((folder / 'end.json').read_text())['end_epoch'] - json.loads((folder / 'command.json').read_text())['start_epoch']
        if kernels:
            item['dram_pct_range'] = [min(k['metrics'][DRAM] for k in kernels), max(k['metrics'][DRAM] for k in kernels)]
            item['sm_pct_range'] = [min(k['metrics'][SM] for k in kernels), max(k['metrics'][SM] for k in kernels)]
            item['strong_indicator_count'] = sum(k['strong_indicator'] for k in kernels)
        captures.append(item)
    result = {'captures': captures, 'total_kernels': sum(len(c['kernels']) for c in captures),
              'capture_wall_minutes': sum(c.get('wall_seconds', 0) for c in captures) / 60,
              'limits': 'Cold-cache kernel replay, unlocked clocks, selected early GEMV samples; not normal API timing or a causal optimization comparison.'}
    result['inferred_projection_groups'] = {}
    context = json.loads((OUT/'trace-context.json').read_text())['first_eight_gemv']
    for capture in captures:
        for k in capture['kernels']:
            prior = context[int(k['id'])]
            assert k['grid'] == str(tuple(prior['grid'])) and k['block'] == str(tuple(prior['block']))
            dims = prior['mm_input_dims'][1]
            k['inferred_weight_bytes'] = dims[0] * dims[1] * 2
            k['read_to_inferred_weight_bytes_ratio'] = k['metrics'][READ] / k['inferred_weight_bytes']
    for name, ids in {'vocabulary_after_prefill': ['0'], 'decode_qkv': ['1', '5'],
                      'decode_attention_output': ['2', '6'], 'decode_gate_up': ['3', '7'],
                      'decode_down': ['4']}.items():
        group = [k for c in captures for k in c['kernels'] if k['id'] in ids]
        if not group:
            continue
        result['inferred_projection_groups'][name] = {
            'count': len(group), 'strong_indicator_count': sum(k['strong_indicator'] for k in group),
            'dram_pct_range': [min(k['metrics'][DRAM] for k in group), max(k['metrics'][DRAM] for k in group)],
            'sm_pct_range': [min(k['metrics'][SM] for k in group), max(k['metrics'][SM] for k in group)],
            'read_GB_per_s_range': [min(k['read_GB_per_s'] for k in group), max(k['read_GB_per_s'] for k in group)]}
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({**result, 'captures': [{k:v for k,v in c.items() if k != 'kernels'} for c in captures]}, indent=2))


if __name__ == '__main__':
    main()
