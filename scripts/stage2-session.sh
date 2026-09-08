#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/Lunacia/Inference
export STAGE1_CONFIG="$PWD/configs/stage2-cap4.env"
export STAGE1_STATE_DIR="$PWD/.tmp/stage2"
py="$HOME/.venvs/inference-vllm/bin/python"
"$py" scripts/stage2-prepare.py
"$py" -m unittest discover -s tests -v > artifacts/stage2/regression.log 2>&1
"$py" -c 'import json,time; from pathlib import Path; t=time.time(); Path("artifacts/stage2/budget.json").write_text(json.dumps({"start_epoch":t,"deadline_epoch":t+5400,"accounting":"conservative wall time including startup and idle"},indent=2))'
cleanup() {
    bash scripts/stop-vllm-baseline.sh > artifacts/stage2/cleanup.log 2>&1 || true
    nvidia-smi >> artifacts/stage2/cleanup.log
    ss -ltnp >> artifacts/stage2/cleanup.log
}
trap cleanup EXIT
bash scripts/start-vllm-baseline.sh > artifacts/stage2/start.log 2>&1
for _ in $(seq 1 120); do
    if curl -sf --max-time 2 http://127.0.0.1:8000/health >/dev/null; then break; fi
    sleep 1
done
curl -sf --max-time 2 http://127.0.0.1:8000/health >/dev/null
cp .tmp/stage2/vllm-server.log artifacts/stage2/server-startup.log
"$py" scripts/stage2-run.py anchor1 512 128 4
"$py" scripts/stage2-run.py anchor2 512 128 4
"$py" - <<'PY'
import json
from pathlib import Path
p=Path('artifacts/stage2')
a,b=[json.loads((p/f'anchor{i}/summary.json').read_text())['output_throughput_tok_s'] for i in (1,2)]
d=abs(b-a)/a*100
(p/'readiness.json').write_text(json.dumps({'anchor1':a,'anchor2':b,'difference_percent':d,'passed':d<=5},indent=2))
if d>5: raise SystemExit('STOP: anchor drift exceeds 5%; investigate device state')
PY
echo 'READY: waiting for bounded next-step commands'
# Keep WSL alive; the deadline is also an independent cleanup watchdog.
while [[ ! -f artifacts/stage2/finish-session ]]; do
    "$py" -c 'import json,time; from pathlib import Path; raise SystemExit(time.time()>=json.loads(Path("artifacts/stage2/budget.json").read_text())["deadline_epoch"])' || break
    sleep 2
done
cp .tmp/stage2/vllm-server.log artifacts/stage2/server-final.log
