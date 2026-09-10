"""Serve and query the existing local vLLM API with a reusable document prefix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import socket
import time

from stage3_lib import ROOT, OwnedVllm, chat_request, now_utc

CONFIG = ROOT / 'configs/local-document-qa.json'


def configuration(prefix: bool = True) -> dict:
    return {**json.loads(CONFIG.read_text(encoding='utf-8')), 'prefix_caching': prefix}


def document_prompt(document: str, question: str) -> str:
    # Do not insert question, timestamp, request ID or changing history before
    # the document. Identical document bytes retain the longest shared prefix.
    return ('请只根据以下文档回答末尾的问题。没有依据时说明文档未提供。'
            '简洁回答，保留所需事实，不要展示思考过程。\n\n文档：\n'
            + document + '\n\n问题：' + question)


def model_name(config: dict) -> str:
    return 'stage3-' + config['label'].lower()


def port_open(config: dict) -> bool:
    with socket.socket() as sock:
        sock.settimeout(2)
        return sock.connect_ex((config['host'], int(config['port']))) == 0


def ask(config: dict, document: str, question: str, stream: bool = True) -> dict:
    return chat_request(f"http://{config['host']}:{config['port']}",
                        document_prompt(document, question), model_name(config),
                        128, stream=stream, timeout=60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    serve = sub.add_parser('serve')
    serve.add_argument('--no-prefix-cache', action='store_true')
    serve.add_argument('--cpu-kv-cache', action='store_true',
                       help='Use the optional 8 GiB CPU KV cache profile (pinned WSL runtime)')
    query = sub.add_parser('ask')
    query.add_argument('--document', type=Path, required=True)
    query.add_argument('--question', required=True)
    query.add_argument('--nonstream', action='store_true')
    args = parser.parse_args()
    config = configuration(not getattr(args, 'no_prefix_cache', False))
    if args.command == 'ask':
        result = ask(config, args.document.read_text(encoding='utf-8'),
                     args.question, not args.nonstream)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result.get('stream_complete') or result.get('finish_reason') != 'stop':
            raise SystemExit(1)
        return
    if port_open(config):
        raise SystemExit('Port 8000 is occupied; refusing to replace another service')
    server_type = OwnedVllm
    if args.cpu_kv_cache:
        if args.no_prefix_cache:
            parser.error('--cpu-kv-cache requires prefix caching')
        from cpu_kv_api import CpuKvServer
        server_type = CpuKvServer
    server = server_type(config, ROOT / '.tmp/document-qa' / now_utc())
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        server.start(timeout=180)
        print('Ready at http://127.0.0.1:8000/v1/chat/completions', flush=True)
        print('Use ask in another terminal; Ctrl+C stops this owned service.', flush=True)
        while server.process.poll() is None:
            time.sleep(1)
        raise SystemExit('vLLM exited unexpectedly; inspect .tmp/document-qa logs')
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == '__main__':
    main()
