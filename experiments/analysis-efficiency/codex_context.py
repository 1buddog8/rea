"""Retain owned Codex tool/usage observations without account or ambient state."""

import hashlib
import io
import json
import os


_TOOL_ITEM_TYPES = frozenset((
    'custom_tool_call', 'custom_tool_call_output',
    'function_call', 'function_call_output',
))


def token_counts(value):
    if not isinstance(value, dict):
        raise ValueError('Expected a token-count object')
    result = {}
    for key in ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens',
                'output_tokens', 'reasoning_output_tokens', 'total_tokens'):
        if key in value:
            count = value[key]
            if (type(count) is not int and (not isinstance(count, int) or isinstance(count, bool))
                    or count < 0):
                raise ValueError('Expected nonnegative integer token counts')
            result[key] = count
    return result


def _select_token_counts(selected, source, keys):
    """Store token_counts(source[key]) in selected for each present key, in one call."""
    exact_type, integer = type, int
    for key in keys:
        if key in source:
            value = source[key]
            if not isinstance(value, dict):
                raise ValueError('Expected a token-count object')
            result = {}
            for name in ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens',
                         'output_tokens', 'reasoning_output_tokens', 'total_tokens'):
                if name in value:
                    count = value[name]
                    if (exact_type(count) is not integer
                            and (not isinstance(count, int) or isinstance(count, bool))
                            or count < 0):
                        raise ValueError('Expected nonnegative integer token counts')
                    result[name] = count
            selected[key] = result


def selected_event(event):
    """Select observed items; these are not the exact submitted model request."""
    if not isinstance(event, dict):
        raise ValueError('Expected an event object with an object payload')
    payload = event.get('payload')
    if not isinstance(payload, dict):
        raise ValueError('Expected an event object with an object payload')
    kind = event.get('type')
    if not isinstance(kind, str):
        raise ValueError('Expected an event type string')
    timestamp = event.get('timestamp')
    if timestamp is not None and not isinstance(timestamp, str):
        raise ValueError('Expected a timestamp string or omission')
    if kind == 'response_item':
        try:
            if payload.get('type') not in _TOOL_ITEM_TYPES:
                return None
        except TypeError:  # unhashable values are never tool item types
            return None
        selected = {}
        for key in ('type', 'call_id', 'name', 'input', 'arguments', 'output'):
            if key in payload:
                selected[key] = payload[key]
        return {'timestamp': timestamp, 'type': kind, 'payload': selected}
    if kind == 'token_usage_record':
        selected = {}
        if 'response_id' in payload:
            selected['response_id'] = payload['response_id']
        _select_token_counts(selected, payload, ('usage', 'turn_token_usage', 'thread_token_usage'))
        return {'timestamp': timestamp, 'type': kind, 'payload': selected}
    if kind == 'event_msg' and payload.get('type') == 'token_count':
        info = payload.get('info')
        if info is not None:
            if not isinstance(info, dict):
                raise ValueError('Expected token-count info')
            selected_info = {}
            _select_token_counts(selected_info, info, ('total_token_usage', 'last_token_usage'))
            if 'model_context_window' in info:
                window = info['model_context_window']
                if window is not None and (not isinstance(window, int) or isinstance(window, bool) or window <= 0):
                    raise ValueError('Expected a positive model context window')
                selected_info['model_context_window'] = window
            info = selected_info
        return {'timestamp': timestamp, 'type': kind, 'payload': {'type': 'token_count', 'info': info}}
    return None


def _event_encoder():
    """Return iterencode(value, 0) whose joined chunks equal
    JSONEncoder(ensure_ascii=False).encode for objects, building the C encoder
    once instead of on every call. Values decoded from JSON text are never
    circular, so the encoder skips circular-reference markers."""
    encoder = json.JSONEncoder(ensure_ascii=False, check_circular=False)
    make = json.encoder.c_make_encoder
    if make is not None:
        args = (encoder.default, json.encoder.encode_basestring, None)
        tail = (encoder.key_separator, encoder.item_separator,
                encoder.sort_keys, encoder.skipkeys, encoder.allow_nan)
        for extra in ((), (float.__repr__,)):
            try:
                return make(None, *args, *extra, *tail)
            except TypeError:
                continue
    return lambda value, _level: (encoder.encode(value),)


def _indented_json(value, level=0):
    """Return json.dumps(value, indent=2) for the str-keyed receipt values,
    formatting containers directly instead of through the pure-Python encoder."""
    kind = type(value)
    if kind is str:
        return json.encoder.encode_basestring_ascii(value)
    if kind is int:
        return int.__repr__(value)
    if kind is dict or kind is list:
        if not value:
            return '{}' if kind is dict else '[]'
        inner = '\n' + '  ' * (level + 1)
        if kind is dict:
            quote = json.encoder.encode_basestring_ascii
            items = [quote(key) + ': ' + _indented_json(item, level + 1)
                     for key, item in value.items()]
            return '{' + inner + (',' + inner).join(items) + '\n' + '  ' * level + '}'
        items = [_indented_json(item, level + 1) for item in value]
        return '[' + inner + (',' + inner).join(items) + '\n' + '  ' * level + ']'
    return json.dumps(value)


def capture_tool_context(codex_home, output):
    """Capture only an explicitly selected trial's rollouts; never its auth/config."""
    sessions = codex_home / 'sessions'
    blocked = sessions.is_symlink()
    sources = [] if blocked else sorted(sessions.rglob('*.jsonl'))
    target = output / 'codex-tool-context.jsonl'
    source_receipts = []
    errors = [{'source': 'sessions', 'reason': 'not_an_owned_directory'}] if blocked else []
    retained = 0
    iterencode = _event_encoder()
    join = ''.join
    scan_once = json.scanner.make_scanner(json.JSONDecoder())
    with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                   'w', encoding='utf-8') as result:
        for source in sources:
            name = source.relative_to(codex_home).as_posix()
            if source.is_symlink() or not source.is_file():
                errors.append({'source': name, 'reason': 'not_an_owned_regular_file'})
                continue
            with source.open('rb') as handle:
                data = handle.read()
            digest = hashlib.sha256(data)
            counts = {}
            # Decoding, selection and encoding depend only on the line bytes,
            # so identical lines within one source are processed once.
            outcomes = {}
            with io.BytesIO(data) as lines:
                for ordinal, line in enumerate(lines, 1):
                    outcome = outcomes.get(line)
                    if outcome is not None:
                        reason, kind, encoded = outcome
                        if reason is not None:
                            errors.append({'source': name, 'line': ordinal,
                                           'reason': reason})
                            continue
                        counts[kind] = counts.get(kind, 0) + 1
                        if encoded is not None:
                            result.write(encoded)
                            retained += 1
                        continue
                    try:
                        # A leading '{' followed by a nonzero byte is what
                        # json.detect_encoding resolves to plain UTF-8.
                        if line[:1] == b'{' and line[1:2] != b'\x00':
                            text = line.decode('utf-8', 'surrogatepass')
                            event, end = scan_once(text, 0)
                            if text[end:].strip(' \t\n\r'):
                                raise json.JSONDecodeError('Extra data', text, end)
                        else:
                            event = json.loads(line)
                        selected = selected_event(event)
                    except ValueError as error:
                        reason = type(error).__name__
                        outcomes[line] = (reason, None, None)
                        errors.append({'source': name, 'line': ordinal,
                                       'reason': reason})
                        continue
                    kind = event['type']  # selected_event checked it is a str
                    counts[kind] = counts.get(kind, 0) + 1
                    if selected is not None:
                        encoded = join(iterencode(selected, 0)) + '\n'
                        result.write(encoded)
                        retained += 1
                    else:
                        encoded = None
                    outcomes[line] = (None, kind, encoded)
            source_receipts.append({'source': name, 'sha256': digest.hexdigest(),
                                    'event_counts': counts})
    retained_digest = hashlib.sha256()
    with target.open('rb') as captured:
        for chunk in iter(lambda: captured.read(1024 * 1024), b''):
            retained_digest.update(chunk)
    receipt = {
        'status': 'partial' if errors else 'captured' if source_receipts else 'unavailable',
        'retained_file': target.name,
        'retained_sha256': retained_digest.hexdigest(),
        'retained_events': retained, 'sources': source_receipts, 'errors': errors,
        'scope': 'Owned client tool-call/reply and per-response usage items, not '
                 'the exact submitted model request. Session/account metadata, '
                 'world state, user/developer messages, encrypted reasoning and '
                 'internal passthrough are omitted. Authentication, configuration '
                 'and ambient environment are never copied by this capture.',
    }
    (output / 'context-receipt.json').write_text(_indented_json(receipt) + '\n')
    return receipt
