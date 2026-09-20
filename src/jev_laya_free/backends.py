"""Offline fixtures and optional local-only Laya inference."""
import math
import os
import re
from pathlib import Path
from .schema import answers, require, text


def confidence(probabilities):
    values = list(probabilities.values())
    return 1.0 if len(values) == 1 else max(0.0, 1 + sum(p * math.log(p) for p in values if p) / math.log(len(values)))


class RulesBackend:
    """Lexical-overlap demo, NOT a semantic classifier or safety decision maker."""
    model = 'local-rules-v1'

    def predict(self, state, questions):
        tokens = set(re.findall(r'\w+', text(state).lower()))
        def distribution(labels):
            weights = {k: 1 + len(tokens & set(re.findall(r'\w+', text(v).lower()))) for k, v in labels.items()}
            total = sum(weights.values())
            return {k: v / total for k, v in weights.items()}
        output = {}
        for name, q in questions.items():
            kind = q['type']
            if kind == 'noul':
                criteria = q.get('criteria', {})
                p = distribution({k: criteria.get(k, '') for k in ('false', 'true')})
                output[name] = {'type': kind, 'noul': p['true']}
                continue
            labels = ({k: k + ' ' + (text(v) if v is not None else '') for k, v in q['criteria'].items()}
                      if kind == 'choice' else {str(i): text(v) for i, v in enumerate(q['criteria'])})
            p = distribution(labels)
            a = {'type': kind, 'probabilities': p, 'confidence': confidence(p)}
            if kind == 'choice':
                a['choice'] = max(p, key=p.get)
            else:
                a.update(score=sum(int(k)*v for k, v in p.items()), legend=labels)
            output[name] = a
        return {'answers': output, 'usage': {'input_tokens': 0, 'output_tokens': 0}}


class LayaBackend:
    model = 'local-laya'

    def __init__(self, model_path=None, device=None):
        path = Path(model_path or os.environ.get('JEV_LAYA_MODEL_PATH', '')).expanduser()
        require(bool(model_path or os.environ.get('JEV_LAYA_MODEL_PATH')) and path.is_dir(),
                'JEV_LAYA_MODEL_PATH must identify an existing local model directory')
        # Prevent Transformers/Hub from retrieving missing assets.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import laya
        self.agent = laya.load(str(path.resolve()), device=device or os.environ.get('JEV_LAYA_DEVICE', 'cpu'))

    def predict(self, state, questions):
        require(len(questions) <= 8, 'Laya backend permits at most 8 questions')
        require(all(q['type'] != 'choice' or len(q['criteria']) <= 8 for q in questions.values()),
                'Laya backend permits at most 8 choice options')
        translated = {}
        for name, q in questions.items():
            item = {**q, 'instructions': text(q['instructions'])}
            if q['type'] == 'score':
                item['criteria'] = [text(v) for v in q['criteria']]
            translated[name] = item
        raw = self.agent.predict(state, translated)
        # Strip Laya-only action metadata and Noul confidence, preserving Jev wire fields.
        return {'answers': answers(raw['answers'], questions), 'usage': raw['usage']}
