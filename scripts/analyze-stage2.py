"""Analyze observed batches, preserving missing metrics and histogram deltas."""
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/stage2'


def parse_metrics(text):
    result = {}
    for line in text.splitlines():
        if line and not line.startswith('#'):
            key, value, *_ = line.split()
            result[key] = float(value)
    return result


def analyze(folder):
    before = parse_metrics((folder/'metrics-before.txt').read_text())
    after = parse_metrics((folder/'metrics-after.txt').read_text())
    deltas = {k:v-before[k] for k,v in after.items() if k in before and
              (re.search(r'_(sum|count|bucket)(\{|$)', k) or 'preemptions_total' in k)}
    timing = {}
    for metric in ['queue','prefill','decode','inference']:
        prefix = f'vllm:request_{metric}_time_seconds'
        counts = [v for k,v in deltas.items() if k.startswith(prefix+'_count')]
        sums = [v for k,v in deltas.items() if k.startswith(prefix+'_sum')]
        timing[metric] = {'count':sum(counts), 'mean_seconds':sum(sums)/sum(counts) if counts and sum(counts)>0 else None}
    samples = [json.loads(s) for s in (folder/'telemetry.jsonl').read_text().splitlines()]
    requests = [json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
    for r in requests:
        r['derived_tpot_seconds'] = (r['latency']-r['ttft'])/(r['output_tokens']-1) if r['success'] and r['output_tokens']>1 else None
    (folder/'request-timings.json').write_text(json.dumps(requests,indent=2))
    gauges = {}
    for metric in ['num_requests_running','num_requests_waiting','kv_cache_usage_perc']:
        values = [v for s in samples for k,v in parse_metrics(s.get('metrics','')).items() if k.startswith('vllm:'+metric+'{')]
        gauges[metric] = {'max': max(values) if values else None, 'samples':len(values)}
    gpu = []
    for s in samples:
        try:
            fields=s['gpu'].split(',')
            gpu.append([float(v.strip()) for v in fields[:7]])
        except (ValueError, AttributeError):
            pass
    gpu_ranges = {name: {'min':min(v[i] for v in gpu),'max':max(v[i] for v in gpu)} for i,name in enumerate(['temperature_c','power_w','sm_mhz','memory_mhz','gpu_util','memory_util','memory_mib'])} if gpu else None
    cpu_percent=[]
    for a,b in zip(samples,samples[1:]):
        av=list(map(int,a['proc_stat'].split()[1:9]));bv=list(map(int,b['proc_stat'].split()[1:9]))
        delta=[y-x for x,y in zip(av,bv)];total=sum(delta)
        if total>0:cpu_percent.append(100*(total-delta[3]-delta[4])/total)
    available=[int(re.search(r'MemAvailable:\s+(\d+)',s['meminfo'])[1])/1024 for s in samples]
    host={'scope':'WSL Linux VM aggregate CPU, not per-core Windows host utilization',
          'cpu_percent_mean':statistics.mean(cpu_percent) if cpu_percent else None,
          'cpu_percent_max':max(cpu_percent) if cpu_percent else None,
          'mem_available_mib_min':min(available) if available else None}
    result={'timing_including_warmups':timing,'histogram_deltas':deltas,'sampled_gauges':gauges,'gpu_ranges_all_batch_samples':gpu_ranges,'host':host,
            'note':'Server deltas include warmups + measurement. Client measured aggregates exclude warmups. Initial probe is disabled by the pinned CLI default. Gauges are sampled, not exhaustive peaks.'}
    (folder/'observations.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    rows=[]
    for p in sorted(OUT.glob('*/summary.json')):
        row=json.loads(p.read_text()); row['batch']=p.parent.name
        row['observations']=analyze(p.parent)
        rows.append(row)
    knees=[]
    for workload in ['A','B','C']:
        points=sorted([r for r in rows if r['batch'].startswith('sweep-'+workload+'-')],key=lambda r:r['concurrency'])
        for a,b in zip(points,points[1:]):
            gain=b['output_throughput_tok_s']/a['output_throughput_tok_s']-1
            rise=max(b[k]/a[k]-1 for k in ['p95_ttft_ms','p95_e2el_ms'])
            knees.append({'workload':workload,'from':a['concurrency'],'to':b['concurrency'],'throughput_gain':gain,'max_p95_rise':rise,'flag':gain<.1 and rise>.25})
    (OUT/'analysis.json').write_text(json.dumps({'rows':rows,'knees':knees},indent=2))
    print(json.dumps({'batches':len(rows),'knees':knees},indent=2))


if __name__=='__main__': main()
