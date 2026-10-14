"""Observe a finished payload while its owned cgroup observer is still alive."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

_DECODER = json.JSONDecoder()
_SCAN = _DECODER.scan_once
_WHITESPACE = json.decoder.WHITESPACE.match
_OBSERVATION_NAMES = ('phase', 'ownership_token', 'payload_exit_code', 'observer_pid',
                      'remaining_processes', 'payload_cleanup_verified',
                      'cpu_stat', 'memory_peak_bytes', 'control_group')


_COUNTERS = {}


def pairs(path):
    text = path.read_text()
    # Parsing depends only on the counter text; reuse valid results for repeated reads.
    cached = _COUNTERS.get(text)
    if cached is not None:
        return dict(cached)
    values = {}
    for line in text.splitlines():
        name, value = line.split()
        number = int(value)
        if number < 0 or name in values:
            raise ValueError('Invalid kernel counter')
        values[name] = number
    if len(_COUNTERS) >= 1024:
        _COUNTERS.clear()
    _COUNTERS[text] = dict(values)
    return values


def snapshot(group, token, code):
    actors = [int(line) for line in (group / 'cgroup.procs').read_text().splitlines()]
    observer = os.getpid()
    return {
        'phase': 'payload-exited-observer-alive', 'ownership_token': token,
        'payload_exit_code': code, 'observer_pid': observer,
        'remaining_processes': actors, 'payload_cleanup_verified': actors == [observer],
        'resource_events': {name: pairs(group / name)
                            for name in ['memory.events', 'pids.events']},
        'cpu_stat': pairs(group / 'cpu.stat'),
        'memory_peak_bytes': int((group / 'memory.peak').read_text()),
        'control_group': str(group.relative_to('/sys/fs/cgroup')),
    }


_RECEIPTS = {}
_RECEIPT_LIMIT = 1024


def read_observation(path, token):
    """Validate the observer receipt; missing/malformed witnesses stay unknown."""
    try:
        if path.is_symlink():
            return None
        text = path.read_text()
        # Validation depends only on the receipt text; reuse it for repeated reads.
        fields = _RECEIPTS.get(text, False)
        if fields is False:
            fields = _validated(text)
            if len(_RECEIPTS) >= _RECEIPT_LIMIT:
                _RECEIPTS.clear()
            _RECEIPTS[text] = fields
        if fields is None:
            return None
        owner, phase, code, pid, actors, verified, cpu, peak, group, memory, tasks = fields
        if owner != token:
            return None
        return {
            'phase': phase, 'ownership_token': owner,
            'payload_exit_code': code, 'observer_pid': pid, 'remaining_processes': list(actors),
            'payload_cleanup_verified': verified,
            'cpu_stat': dict(cpu), 'memory_peak_bytes': peak, 'control_group': group,
            'resource_events': {'memory.events': dict(memory), 'pids.events': dict(tasks)},
        }
    except (OSError, ValueError, TypeError, StopIteration, KeyError):
        return None


def _validated(text):
    """Return the checked receipt fields, or None when the receipt is malformed."""
    try:
        # Same acceptance as _DECODER.decode without its two Python-level frames.
        # Skip the regex when there is no surrounding whitespace to match.
        value, end = _SCAN(text, 0 if text[:1] not in ' \t\n\r' else _WHITESPACE(text, 0).end())
        size = len(text)
        if end != size and _WHITESPACE(text, end).end() != size:
            return None
        # Decoded JSON only yields exact dict, list, str, int and bool types.
        if type(value) is not dict:
            return None
        owner = value.get('ownership_token')
        # Every remaining field is required; a missing key raises KeyError.
        phase = value['phase']
        if phase != 'payload-exited-observer-alive':
            return None
        group = value['control_group']
        if type(group) is not str or not group:
            return None
        code = value['payload_exit_code']
        pid = value['observer_pid']
        peak = value['memory_peak_bytes']
        if type(code) is not int or type(pid) is not int or type(peak) is not int:
            return None
        if pid <= 0 or peak < 0:
            return None
        actors = value['remaining_processes']
        if type(actors) is not list:
            return None
        for x in actors:
            if type(x) is not int or x <= 0:
                return None
        verified = value['payload_cleanup_verified']
        if type(verified) is not bool or verified != (actors == [pid]):
            return None
        events = value['resource_events']
        if type(events) is not dict:
            return None
        memory = events['memory.events']
        if (type(memory) is not dict or 'high' not in memory or 'max' not in memory
                or 'oom' not in memory or 'oom_kill' not in memory):
            return None
        for x in memory.values():
            if type(x) is not int or x < 0:
                return None
        tasks = events['pids.events']
        if type(tasks) is not dict or 'max' not in tasks:
            return None
        for x in tasks.values():
            if type(x) is not int or x < 0:
                return None
        cpu = value['cpu_stat']
        if type(cpu) is not dict or 'usage_usec' not in cpu:
            return None
        for x in cpu.values():
            if type(x) is not int or x < 0:
                return None
        return owner, phase, code, pid, actors, verified, cpu, peak, group, memory, tasks
    except (ValueError, TypeError, StopIteration, KeyError):
        return None


def main():
    output, release, token = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    command = sys.argv[4:]
    if not command:
        raise SystemExit('Missing explicit payload command')
    child = subprocess.Popen(command)
    code = child.wait()
    paths = [line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
             if line.startswith('0::')]
    try:
        if len(paths) != 1:
            raise ValueError('Expected one unified cgroup')
        group = Path('/sys/fs/cgroup') / paths[0].lstrip('/')
        observation = snapshot(group, token, code)
        with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                       'w') as file:
            json.dump(observation, file)
        # Keep the owner alive until its supervisor acknowledges the receipt.
        # Bound a broken supervisor by the existing ten-second cleanup interval.
        deadline = time.monotonic() + 10
        while not release.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
    except (OSError, ValueError) as error:
        print('Payload counter observation unavailable: ' + type(error).__name__, file=sys.stderr)
    raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == '__main__':
    main()
