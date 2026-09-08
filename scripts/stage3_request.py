"""Disposable HTTP worker; the parent kills and reaps it at the deadline."""
import json
import sys
from stage3_lib import _stream_chat, _nonstream_chat

if __name__ == "__main__":
    args = json.loads(sys.stdin.read())
    request = _stream_chat if args.get("stream", True) else _nonstream_chat
    result = request(args["base_url"], args["body"], args["timeout"])
    sys.stdout.write(json.dumps(result, ensure_ascii=True))
