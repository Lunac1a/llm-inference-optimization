#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/Lunacia/Inference
export STAGE1_CONFIG="$PWD/configs/stage2-cap4.env"
export STAGE1_STATE_DIR="$PWD/.tmp/stage2"
py="$HOME/.venvs/inference-vllm/bin/python"
cleanup() {
    cp .tmp/stage2/vllm-server.log artifacts/stage2/server-recovery-fixed-final.log || true
    bash scripts/stop-vllm-baseline.sh > artifacts/stage2/cleanup-recovery-fixed.log 2>&1 || true
    nvidia-smi >> artifacts/stage2/cleanup-recovery-fixed.log
    ss -ltnp >> artifacts/stage2/cleanup-recovery-fixed.log
}
trap cleanup EXIT
bash scripts/start-vllm-baseline.sh > artifacts/stage2/start-recovery-fixed.log 2>&1
for _ in $(seq 1 120); do
    if curl -sf --max-time 2 http://127.0.0.1:8000/health >/dev/null; then break; fi
    sleep 1
done
curl -sf --max-time 2 http://127.0.0.1:8000/health >/dev/null
# One predeclared recovery only; continuous work bounded by a 180s batch deadline.
timeout 180 "$py" scripts/stage2-run.py recovery-fixed-warmup 512 128 4 --count 128
"$py" scripts/stage2-run.py recovery-fixed-anchor1 512 128 4
"$py" scripts/stage2-run.py recovery-fixed-anchor2 512 128 4
"$py" - <<'PY'
import json
from pathlib import Path
p=Path('artifacts/stage2')
a,b=[json.loads((p/f'recovery-fixed-anchor{i}/summary.json').read_text())['output_throughput_tok_s'] for i in (1,2)]
d=abs(b-a)/a*100
(p/'readiness-recovery.json').write_text(json.dumps({'anchor1':a,'anchor2':b,'difference_percent':d,'passed':d<=5},indent=2))
if d>5: raise SystemExit('STOP: recovery anchor drift; environment blocker, no more readiness attempts')
PY
echo 'READY: waiting for bounded next-step commands'
while [[ ! -f artifacts/stage2/finish-session ]]; do
    "$py" -c 'import json,time; from pathlib import Path; raise SystemExit(time.time()>=json.loads(Path("artifacts/stage2/budget.json").read_text())["deadline_epoch"])' || break
    sleep 2
done
