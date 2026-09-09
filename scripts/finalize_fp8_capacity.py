"""Verify preservation and export compact measurements after collection."""
import hashlib
import json
from pathlib import Path
from stage3_lib import ROOT, write_json
from validate_fp8_capacity import OUT, hashes
from analyze_fp8_capacity import analyze

def main():
    rows=json.loads((OUT/'measurements.json').read_text(encoding='utf-8'))
    assert len(rows)==8
    analysis=analyze(rows)
    write_json(OUT/'analysis.json',analysis)
    old=json.loads((OUT/'old-evidence-hashes.json').read_text(encoding='utf-8'))
    changed=[p for p,h in old.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    frozen=json.loads((OUT/'protocol.json').read_text())['hashes']
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in frozen.items())
    performance=[r for p in OUT.glob('*/perf-*/requests.json') for r in json.loads(p.read_text(encoding='utf-8'))]
    quality=[r for p in OUT.glob('*/quality-*/requests.json') for r in json.loads(p.read_text(encoding='utf-8'))]
    assert len(performance)==60 and all(r['stream_complete'] and r['output_tokens']==128 for r in performance)
    assert len(quality)==18 and all(r['stream_complete'] and r['finish_reason']=='stop' for r in quality)
    cleanup=json.loads((OUT/'cleanup.json').read_text())
    assert cleanup['port_closed']
    assert cleanup['compute_processes'].strip()=='pid, process_name'
    for row in rows:
        log=(OUT/row['label']/'server/server.log').read_text(encoding='utf-8')
        dtype='fp8_per_token_head' if row['fp8'] else 'bfloat16'
        assert 'Using AttentionBackendEnum.TRITON_ATTN backend.' in log
        assert 'kv_cache_dtype='+dtype in log and 'dtype=torch.bfloat16' in log
        assert 'enable_prefix_caching=True' in log
        stop=json.loads((OUT/row['label']/'server/stop.json').read_text())
        assert stop['returncode']==0
    write_json(OUT/'verification.json',{'old_files_unchanged':len(old),'frozen_sources_unchanged':len(frozen),
        'performance_complete':len(performance),'quality_transport_complete':len(quality),
        'quality_content_review':'answer-review.md','verified_backend_dtype_and_clean_stops':8,
        'cleanup':cleanup,'analysis':analysis})
    lines=['| Run | Wave | Success | TTFT p95 s | E2E p95 s | Output tok/s | Hit/query ratio | Queue total s | Running max | Waiting max | Active KV max |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        for i,w in enumerate(row['waves']):
            g=w['gauges']
            lines.append(f"| {row['label']} | {i} | {w['success']}/{w['requests']} | {w['ttft']['p95']:.3f} | {w['e2e']['p95']:.3f} | {w['throughput']:.2f} | {w['cache']['hit_ratio']:.4%} | {w['deltas']['vllm:request_queue_time_seconds_sum']:.6f} | {g['vllm:num_requests_running']['max']:.0f} | {g['vllm:num_requests_waiting']['max']:.0f} | {g['vllm:kv_cache_usage_perc']['max']:.2%} |")
    (OUT/'measurement-table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(OUT/'evidence-sha256.json',hashes(p for p in OUT.rglob('*') if p.is_file() and p.name!='evidence-sha256.json'))
    print(json.dumps({'preserved':len(old),'analysis':analysis},indent=2))

if __name__=='__main__':
    main()
