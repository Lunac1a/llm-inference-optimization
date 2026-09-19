"""Windows-only temporary power requests; lifetime scoped to the WSL evaluation."""
import ctypes
from ctypes import wintypes
import datetime
import json
import os
from pathlib import Path
import subprocess


def main():
    assert os.name == 'nt'
    root = Path(__file__).resolve().parents[2]
    out = root / '.local/research/ruler-final-2026-09-19/resume-2'
    out.mkdir(parents=True, exist_ok=False)
    class Detailed(ctypes.Structure):
        _fields_ = [('module', wintypes.HANDLE), ('id', wintypes.ULONG), ('count', wintypes.ULONG), ('strings', ctypes.POINTER(wintypes.LPWSTR))]
    class Reason(ctypes.Union):
        _fields_ = [('simple', wintypes.LPWSTR), ('detailed', Detailed)]
    class Context(ctypes.Structure):
        _fields_ = [('version', wintypes.ULONG), ('flags', wintypes.DWORD), ('reason', Reason)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.PowerCreateRequest.argtypes = [ctypes.POINTER(Context)]; kernel.PowerCreateRequest.restype = wintypes.HANDLE
    kernel.PowerSetRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]; kernel.PowerSetRequest.restype = wintypes.BOOL
    kernel.PowerClearRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]; kernel.PowerClearRequest.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]; kernel.CloseHandle.restype = wintypes.BOOL
    context = Context(0, 1, Reason(simple='Mixed KV benchmark continuation; temporary request until completion'))
    handle = kernel.PowerCreateRequest(ctypes.byref(context))
    if handle == wintypes.HANDLE(-1).value: raise ctypes.WinError(ctypes.get_last_error())
    held = []; record = {'pid': os.getpid(), 'active': False, 'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def save(): (out / 'power-guard.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    try:
        for kind in [1, 3]:
            if not kernel.PowerSetRequest(handle, kind): raise ctypes.WinError(ctypes.get_last_error())
            held.append(kind)
        record.update(active=True, requests=['SystemRequired', 'ExecutionRequired'], display_required=False); save()
        with (out / 'orchestration.log').open('x', encoding='utf-8') as log:
            result = subprocess.run(['wsl', '-d', 'Ubuntu-24.04', '--', 'bash', '-lc',
                '/home/lunacia/.venvs/inference-vllm/bin/python /mnt/d/Lunacia/Inference/benchmarks/evaluation/resume2.py'],
                stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        record['exit_code'] = result.returncode
    finally:
        for kind in reversed(held): kernel.PowerClearRequest(handle, kind)
        kernel.CloseHandle(handle)
        record.update(active=False, end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); save()
    raise SystemExit(record.get('exit_code', 1))


if __name__ == '__main__': main()
