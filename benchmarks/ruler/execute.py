"""Wait for completed CPU preparation, then run the frozen GPU protocol once."""
import json
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT

started=time.monotonic()
while not (ROOT/'data-manifest.json').exists():
    if time.monotonic()-started>600:raise TimeoutError('CPU preparation did not finish; no GPU launched')
    time.sleep(5)
for args in [('run.py','pilot'),('run.py','formal'),('report.py',)]:
    subprocess.run([sys.executable,str(Path(__file__).with_name(args[0])),*args[1:]],check=True)
print(json.dumps({'event':'evaluation_complete','report':'docs/ruler-results.md'}),flush=True)
