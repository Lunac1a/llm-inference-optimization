"""Verify immutable evidence and export the hybrid experiment's compact results."""
import ast
import hashlib
import json
import re
from pathlib import Path
from stage3_lib import ROOT,write_json
from validate_hybrid_prefill import OUT,hashes
from analyze_hybrid_prefill import analyze

def main():
    rows=json.loads((OUT/'measurements.json').read_text(encoding='utf-8'))
    assert len(rows)==6
    result=analyze(rows)
    write_json(OUT/'analysis.json',result)
    old=json.loads((OUT/'old-evidence-hashes.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in old.items())
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protocol['sources'].items())
    packages=json.loads((OUT/'preflight.json').read_text())['package_files']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in packages.items())
    perf=[r for p in OUT.glob('*/perf-*/requests.json') for r in json.loads(p.read_text())]
    quality=[r for p in OUT.glob('*/quality-*/requests.json') for r in json.loads(p.read_text())]
    assert len(perf)==54 and all(r['stream_complete'] and r['output_tokens']==128 for r in perf)
    assert len(quality)==27 and all(r['stream_complete'] and r['finish_reason']=='stop' for r in quality)
    cleanup=json.loads((OUT/'cleanup.json').read_text())
    assert cleanup['port_closed'] and cleanup['compute_processes'].strip()=='pid, process_name'
    numerical=json.loads((OUT/'numerical-checks.json').read_text())
    assert all(c['pass'] for c in numerical['comparisons']) and numerical['cache_bytes_unchanged']
    servers=[]
    for row in rows:
        folder=OUT/row['label']
        log=(folder/'server/server.log').read_text()
        dtype='bfloat16' if row['arm']=='bf16' else 'fp8_per_token_head'
        assert 'kv_cache_dtype='+dtype in log and 'dtype=torch.bfloat16' in log
        assert 'enable_chunked_prefill=False' in log and 'enable_prefix_caching=True' in log
        assert 'Using AttentionBackendEnum.TRITON_ATTN backend.' in log
        launch=json.loads((folder/'actual-launch.json').read_text())
        assert '--no-enable-chunked-prefill' in launch['argv'] and '--enable-chunked-prefill' not in launch['argv']
        assert json.loads((folder/'server/stop.json').read_text())['returncode']==0
        cold_layers=set(re.findall(r'HYBRID_FORWARD cold_long_bf16 layer=([^\n]+)',log))
        cached_layers=set(re.findall(r'HYBRID_FORWARD cached_fp8 layer=([^\n]+)',log))
        cold_sequences=[]
        for match in re.findall(r'HYBRID_SCHEDULE (\[.*\])',log):
            cold_sequences.extend(r for r in ast.literal_eval(match) if r[1]-r[0]>=8192 and r[3])
        if row['arm']=='hybrid':
            assert len(cold_layers)==36 and len(cached_layers)==36
            assert len(cold_sequences)==3 and all(r[1]-r[0]==r[2]==16446 for r in cold_sequences)
            assert launch['environment']['VLLM_PLUGINS']=='inference_hybrid_prefill'
        else:
            assert 'HYBRID_REGISTER' not in log and launch['environment']['VLLM_PLUGINS']==''
        tokens=int(re.search(r'GPU KV cache size: ([\d,]+) tokens',log).group(1).replace(',',''))
        servers.append({'run':row['label'],'cache_tokens':tokens,
            'cold_bf16_layers':len(cold_layers),'cached_fp8_layers':len(cached_layers),
            'full_cold_sequences':cold_sequences,'stop_returncode':0})
    write_json(OUT/'verification.json',{'old_files_unchanged':len(old),
        'frozen_sources_unchanged':len(protocol['sources']),'installed_sources_unchanged':len(packages),
        'performance_complete':len(perf),'quality_transport_complete':len(quality),
        'quality_review':'answer-review.md','numerical':numerical,'servers':servers,
        'cleanup':cleanup,'analysis':result,'environment_recovery':'environment-attempt/'})
    lines=['| Run | Wave | Success | TTFT p50 / p95 s | E2E p50 / p95 s | Output tok/s | Hit/query | Queue total s | Running max | Waiting max | Active KV peak |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        for i,w in enumerate(row['waves']):
            g=w['gauges']
            lines.append(f"| {row['label']} | {i} | {w['success']}/{w['requests']} | {w['ttft']['p50']:.3f} / {w['ttft']['p95']:.3f} | {w['e2e']['p50']:.3f} / {w['e2e']['p95']:.3f} | {w['throughput']:.2f} | {w['cache']['hit_ratio']:.4%} | {w['deltas']['vllm:request_queue_time_seconds_sum']:.6f} | {g['vllm:num_requests_running']['max']:.0f} | {g['vllm:num_requests_waiting']['max']:.0f} | {g['vllm:kv_cache_usage_perc']['max']:.2%} |")
    (OUT/'measurement-table.md').write_text('\n'.join(lines)+'\n')
    write_json(OUT/'evidence-sha256.json',hashes(p for p in OUT.rglob('*') if p.is_file() and p.name!='evidence-sha256.json'))
    print(json.dumps({'old_files_unchanged':len(old),'servers':servers,'analysis':result},indent=2))

if __name__=='__main__':
    main()
