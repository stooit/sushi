#!/usr/bin/env python3
# sushi-test: tier=live boot=api
"""Check independent DeepSeek streams, cached continuations and cancellation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
import sushi_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=os.environ.get('SUSHI_MODEL'))
    parser.add_argument('--output', default=os.environ.get('SUSHI_RUN_DIR'))
    args = parser.parse_args()
    nonce = sushi_test.attach()
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('SUSHI_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['SUSHI_API_KEY']
    if not args.model:
        with urlopen(Request(sushi_test.url() + '/v1/models', headers=headers), timeout=30) as response:
            args.model = json.load(response)['data'][0]['id']
    if os.environ.get('SUSHI_ARCH', 'deepseek_v41') != 'deepseek_v41':
        sushi_test.skip('DeepSeek request isolation test')
    with urlopen(Request(sushi_test.url() + '/props', headers=headers), timeout=30) as response:
        props = json.load(response)
    if props['settings']['max_concurrent'] < 2 or props['batching']['reason'] == 'exclusive':
        sushi_test.skip('requires resident DeepSeek experts and --max-concurrent 2 or greater')
    output = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix='sushi-dsv41-concurrency-'))
    output.mkdir(parents=True, exist_ok=True)
    results = []

    def stream(body, barrier=None, cancel_after=None):
        if barrier:
            barrier.wait(timeout=30)
        payload = dict(body, model=args.model, stream=True, stream_options={'include_usage': True})
        record = {'first': None, 'end': None, 'done': False, 'finish': None,
                  'usage': {}, 'content': '', 'reasoning': '', 'tools': {}, 'cancelled': False}
        fragments = 0
        req = Request(sushi_test.url() + '/v1/chat/completions',
                      data=json.dumps(payload).encode(), headers=headers)
        with urlopen(req, timeout=900) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    record['done'] = True
                    break
                event = json.loads(data)
                assert 'error' not in event, event.get('error')
                if event.get('usage'):
                    record['usage'] = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    visible = bool(delta.get('content') or delta.get('reasoning_content') or delta.get('tool_calls'))
                    if visible:
                        record['first'] = record['first'] or time.monotonic()
                        fragments += 1
                    record['content'] += delta.get('content') or ''
                    record['reasoning'] += delta.get('reasoning_content') or ''
                    for call in delta.get('tool_calls', []):
                        target = record['tools'].setdefault(call['index'], {'name': '', 'arguments': ''})
                        for field in target:
                            target[field] += call.get('function', {}).get(field) or ''
                    record['finish'] = choice.get('finish_reason') or record['finish']
                if cancel_after and fragments >= cancel_after:
                    record['cancelled'] = True
                    break
        record['end'] = time.monotonic()
        if not record['cancelled']:
            assert record['done'] and record['finish'], 'incomplete SSE response'
            assert record['first'] is not None, 'empty response'
        return record

    def wave(bodies, cancellation=False):
        barrier = threading.Barrier(len(bodies))
        with ThreadPoolExecutor(max_workers=len(bodies)) as pool:
            futures = [pool.submit(stream, body, barrier, 8 if cancellation and i == 0 else None)
                       for i, body in enumerate(bodies)]
            replies = [f.result(timeout=930) for f in futures]
        results.append({'kind': 'cancel' if cancellation else 'concurrent', 'replies': replies})
        assert all(r['first'] is not None for r in replies), 'one stream never generated'
        assert max(r['first'] for r in replies) < min(r['end'] for r in replies), \
            'requests queued to completion instead of interleaving generated output'
        if cancellation:
            assert replies[0]['cancelled'], 'cancellation was not exercised'
            assert replies[1]['done'], 'cancelling one request stopped its neighbour'
        return replies

    shared = nonce + '\n' + '\n'.join(
        f'Requirement {i}: preserve ordering, validate input, handle cancellation and document failure recovery.'
        for i in range(384))
    bodies = [dict(messages=[{'role': 'system', 'content': shared},
                             {'role': 'user', 'content': f'Session {i}: write a detailed implementation plan for these requirements, with at least 30 numbered steps.'}],
                   temperature=0, max_tokens=384, enable_thinking=False)
              for i in range(2)]
    try:
        for mtp in (False, True):
            requests = [dict(body, enable_mtp=mtp) for body in bodies]
            for body in requests:
                results.append({'kind': 'warm', 'mtp': mtp, 'reply': stream(body)})
            replies = wave(requests)
            assert all(r['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) > 0
                       for r in replies), 'warm concurrent requests did not reuse their prompt cache'
        wave([dict(body, enable_mtp=True) for body in bodies], cancellation=True)

        requests = []
        for i in range(2):
            marker = f'{nonce}-session-{i}'
            requests.append(dict(messages=[{'role': 'user', 'content': f'Call record_session with session_id {marker}.'}],
                                 temperature=0, max_tokens=128, enable_thinking=False, enable_mtp=True,
                                 tools=[{'type': 'function', 'function': {'name': 'record_session',
                                         'parameters': {'type': 'object', 'properties': {
                                             'session_id': {'type': 'string', 'enum': [marker]}},
                                             'required': ['session_id'], 'additionalProperties': False}}}],
                                 tool_choice={'type': 'function', 'function': {'name': 'record_session'}}))
        # Short calls check isolation and recovery; overlap is checked by the long streams above.
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(stream, body, barrier) for body in requests]
            replies = [f.result(timeout=930) for f in futures]
        results.append({'kind': 'tool-isolation', 'replies': replies})
        for i, reply in enumerate(replies):
            assert reply['finish'] == 'tool_calls' and len(reply['tools']) == 1
            call = next(iter(reply['tools'].values()))
            assert call['name'] == 'record_session'
            assert json.loads(call['arguments']) == {'session_id': f'{nonce}-session-{i}'}, 'cross-session tool state'
        log = sushi_test.log_since()
        assert '[spec-stats] mode=dspark attempts=' in log, 'DSpark did not engage'
        (output / 'dsv41-concurrency.log').write_text(log)
        results.append({'passed': True})
    finally:
        (output / 'dsv41-concurrency.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
