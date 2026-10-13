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
_REQUIRED_EVENTS = (('memory.events', ('high', 'max', 'oom', 'oom_kill')),
                    ('pids.events', ('max',)))
_OBSERVATION_NAMES = ('phase', 'ownership_token', 'payload_exit_code', 'observer_pid',
                      'remaining_processes', 'payload_cleanup_verified',
                      'cpu_stat', 'memory_peak_bytes', 'control_group')


def pairs(path):
    values = {}
    for line in path.read_text().splitlines():
        name, value = line.split()
        number = int(value)
        if number < 0 or name in values:
            raise ValueError('Invalid kernel counter')
        values[name] = number
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


def read_observation(path, token):
    """Validate the observer receipt; missing/malformed witnesses stay unknown."""
    try:
        if path.is_symlink():
            return None
        # Same acceptance as _DECODER.decode without its two Python-level frames.
        text = path.read_text()
        value, end = _SCAN(text, _WHITESPACE(text, 0).end())
        if _WHITESPACE(text, end).end() != len(text):
            return None
        if not isinstance(value, dict) or value.get('ownership_token') != token:
            return None
        if value.get('phase') != 'payload-exited-observer-alive':
            return None
        if not isinstance(value.get('control_group'), str) or not value['control_group']:
            return None
        for name in ('payload_exit_code', 'observer_pid', 'memory_peak_bytes'):
            if type(value.get(name)) is not int:
                return None
        if value['observer_pid'] <= 0 or value['memory_peak_bytes'] < 0:
            return None
        actors = value.get('remaining_processes')
        if not isinstance(actors, list):
            return None
        for x in actors:
            if type(x) is not int or x <= 0:
                return None
        if type(value.get('payload_cleanup_verified')) is not bool:
            return None
        if value['payload_cleanup_verified'] != (actors == [value['observer_pid']]):
            return None
        events = value.get('resource_events')
        if not isinstance(events, dict):
            return None
        for name, keys in _REQUIRED_EVENTS:
            record = events.get(name)
            if not isinstance(record, dict):
                return None
            for key in keys:
                if key not in record:
                    return None
            for x in record.values():
                if type(x) is not int or x < 0:
                    return None
        cpu = value.get('cpu_stat')
        if not isinstance(cpu, dict) or 'usage_usec' not in cpu:
            return None
        for x in cpu.values():
            if type(x) is not int or x < 0:
                return None
        return {
            'phase': value['phase'], 'ownership_token': value['ownership_token'],
            'payload_exit_code': value['payload_exit_code'],
            'observer_pid': value['observer_pid'], 'remaining_processes': actors,
            'payload_cleanup_verified': value['payload_cleanup_verified'],
            'cpu_stat': cpu, 'memory_peak_bytes': value['memory_peak_bytes'],
            'control_group': value['control_group'],
            'resource_events': {'memory.events': events['memory.events'],
                                'pids.events': events['pids.events']},
        }
    except (OSError, ValueError, TypeError, StopIteration):
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
