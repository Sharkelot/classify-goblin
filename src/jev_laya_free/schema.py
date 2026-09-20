"""Bounded JSON validation and normalization, shared by server and client."""
import json
import math

MAX_BYTES = 65536
MAX_RESPONSE_BYTES = 1048576


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def json_value(value, depth=0):
    require(depth <= 16, 'JSON nesting exceeds 16 levels')
    if isinstance(value, dict):
        require(all(isinstance(k, str) for k in value), 'JSON keys must be strings')
        for v in value.values():
            json_value(v, depth + 1)
    elif isinstance(value, list):
        for v in value:
            json_value(v, depth + 1)
    else:
        require(value is None or type(value) in (str, int, float, bool), 'non-JSON value')
        if type(value) is float:
            require(math.isfinite(value), 'nonfinite JSON number')


def dumps(value):
    json_value(value)
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def loads(data):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    try:
        value = json.loads(data, object_pairs_hook=pairs)
        json_value(value)
        return value
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise ValidationError('invalid JSON') from exc


def text(value):
    return value if isinstance(value, str) else dumps(value)


def description(value, nullable=False):
    require((nullable and value is None) or isinstance(value, (str, dict, list)),
            'description must be string, object, or array')
    require(len(dumps(value).encode()) <= 4096, 'description exceeds 4096 bytes')


def request(body):
    require(isinstance(body, dict), 'request must be an object')
    require(set(body) <= {'model', 'state', 'questions'}, 'unknown request field')
    require({'state', 'questions'} <= set(body), 'state and questions required')
    require(len(dumps(body).encode()) <= MAX_BYTES, 'request exceeds 65536 bytes')
    require(isinstance(body['state'], (str, dict, list)), 'state must be string, object, or array')
    model = body.get('model', 'local-default')
    require(isinstance(model, str) and 0 < len(model) <= 128, 'invalid model')
    questions = body['questions']
    require(isinstance(questions, dict) and 1 <= len(questions) <= 32, 'require 1..32 questions')
    normalized = {}
    for name, q in questions.items():
        require(isinstance(name, str) and 0 < len(name) <= 128, 'invalid question name')
        require(isinstance(q, dict), 'question must be object')
        require(set(q) <= {'type', 'instructions', 'criteria', 'options'}, 'unknown question field')
        kind = q.get('type')
        require(kind in ('choice', 'score', 'noul'), 'unsupported question type')
        require('instructions' in q, 'instructions required')
        description(q['instructions'])
        n = {'type': kind, 'instructions': q['instructions']}
        if kind == 'choice':
            require(('options' in q) != ('criteria' in q), 'choice needs exactly one of options/criteria')
            options = q.get('options', q.get('criteria'))
            if isinstance(options, list):
                require(all(isinstance(k, str) for k in options), 'choice list must contain strings')
                require(len(set(options)) == len(options), 'duplicate option')
                options = dict.fromkeys(options)
            require(isinstance(options, dict) and 1 <= len(options) <= 255, 'require 1..255 options')
            for k, v in options.items():
                require(isinstance(k, str) and 0 < len(k) <= 128, 'invalid option name')
                description(v, nullable=True)
            n['criteria'] = options
        elif kind == 'score':
            require('options' not in q, 'score does not accept options')
            levels = q.get('criteria')
            require(isinstance(levels, list) and 2 <= len(levels) <= 10, 'score needs 2..10 levels')
            for level in levels:
                description(level)
            n['criteria'] = levels
        else:
            require('options' not in q, 'noul does not accept options')
            if 'criteria' in q:
                criteria = q['criteria']
                require(isinstance(criteria, dict) and set(criteria) <= {'true', 'false'}, 'invalid noul criteria')
                for value in criteria.values():
                    description(value)
                n['criteria'] = criteria
        normalized[name] = n
    return {'model': model, 'state': body['state'], 'questions': normalized}


def probability(value):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            'invalid probability/confidence')


def answers(raw, questions):
    require(isinstance(raw, dict) and set(raw) == set(questions), 'answer names mismatch')
    clean = {}
    for name, q in questions.items():
        a = raw[name]
        kind = q['type']
        require(isinstance(a, dict) and a.get('type') == kind, 'answer type mismatch')
        if kind == 'noul':
            probability(a.get('noul'))
            clean[name] = {'type': kind, 'noul': a['noul']}
            continue
        probability(a.get('confidence'))
        keys = list(q['criteria']) if kind == 'choice' else [str(i) for i in range(len(q['criteria']))]
        p = a.get('probabilities')
        require(isinstance(p, dict) and set(p) == set(keys), 'probability keys mismatch')
        for value in p.values():
            probability(value)
        require(abs(sum(p.values()) - 1) <= 0.002, 'probabilities must sum to one')
        clean[name] = {'type': kind, 'confidence': a['confidence'], 'probabilities': p}
        if kind == 'choice':
            require(isinstance(a.get('choice'), str) and a['choice'] in p, 'invalid choice')
            require(p[a['choice']] == max(p.values()), 'choice must maximize probability')
            clean[name]['choice'] = a['choice']
        else:
            score = a.get('score')
            require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= len(keys)-1, 'invalid score')
            require(abs(score - sum(int(k)*p[k] for k in keys)) <= 0.02, 'score must match weighted mean')
            legend = {str(i): text(v) for i, v in enumerate(q['criteria'])}
            require(a.get('legend') == legend, 'score legend mismatch')
            clean[name].update(score=score, legend=legend)
    return clean


def response(body, questions):
    require(isinstance(body, dict), 'response must be object')
    require(isinstance(body.get('model'), str) and bool(body['model']), 'response model required')
    usage = body.get('usage')
    require(isinstance(usage, dict), 'usage required')
    for key in ('input_tokens', 'output_tokens'):
        require(type(usage.get(key)) is int and usage[key] >= 0, 'invalid usage')
    if 'request_id' in body:
        require(isinstance(body['request_id'], str) and bool(body['request_id']), 'invalid request_id')
    answers(body.get('answers'), questions)
    return body
