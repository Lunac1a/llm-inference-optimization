"""Pinned upstream benchmark with a 120s HTTP timeout and exact output timing.

Only this client process is adapted; installed vLLM and Stage 1 are untouched.
"""
import dataclasses
import json
import os
import time

import aiohttp
from vllm.benchmarks import serve
from vllm.entrypoints.cli.main import main

original_session = aiohttp.ClientSession


def bounded_session(*args, **kwargs):
    kwargs['timeout'] = aiohttp.ClientTimeout(total=120)
    return original_session(*args, **kwargs)


aiohttp.ClientSession = bounded_session
original_request = serve.ASYNC_REQUEST_FUNCS['openai']


async def recorded_request(*args, **kwargs):
    output = await original_request(*args, **kwargs)
    with open(os.environ['STAGE2_REQUEST_LOG'], 'a') as handle:
        handle.write(json.dumps({'recorded_at': time.time(), **dataclasses.asdict(output)}) + '\n')
    return output


serve.ASYNC_REQUEST_FUNCS['openai'] = recorded_request
if __name__ == '__main__':
    import sys
    sys.argv[1:1] = ['bench', 'serve']
    main()
