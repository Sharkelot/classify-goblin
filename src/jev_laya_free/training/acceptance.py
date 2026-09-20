"""Offline, advisory acceptance gates. Passing never authorizes execution.

Input is either typed prediction records (question_name, labels, label_index,
probabilities, modality) or paired rows with gold/predictions mappings. Gold
labels take precedence over legacy expected_* annotations and metadata.
"""
from dataclasses import asdict, dataclass
import math
from typing import Mapping

from .metrics import _summary, evaluate_workflow_precedence


@dataclass(frozen=True)
class AcceptanceConfig:
    repeat_recall: float = .95
    repeat_precision: float = .90
    next_hand_macro_f1: float = .85
    code_recall: float = .95
    code_precision: float = .90
    pdf_macro_f1: float = .90
    image_recall: float = .95
    max_ece: float = .10
    min_support: int = 1
    modalities: tuple = ('code', 'pdf', 'image')
    max_latency_p95_ms: float | None = None
    max_vram_mb: float | None = None

    def __post_init__(self):
        for name in ('repeat_recall', 'repeat_precision', 'next_hand_macro_f1',
                     'code_recall', 'code_precision', 'pdf_macro_f1', 'image_recall', 'max_ece'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
        if type(self.min_support) is not int or self.min_support < 1:
            raise ValueError('min_support must be a positive integer')
        if not self.modalities or set(self.modalities) - {'code', 'pdf', 'image'}:
            raise ValueError('modalities must contain code, pdf, or image')
        for name in ('max_latency_p95_ms', 'max_vram_mb'):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f'{name} must be finite and nonnegative')


def _label(value):
    if isinstance(value, Mapping):
        for key in ('label', 'choice', 'value'):
            if key in value:
                return value[key]
        if isinstance(value.get('probabilities'), Mapping):
            return max(value['probabilities'], key=value['probabilities'].get)
    return value


def annotations(record):
    """Resolve semantic labels without inferring repeat risk from unrelated nouls."""
    result = dict(record)
    name = record.get('question_name')
    labels = record.get('labels') or list(record.get('question', {}).get('criteria', {}))
    gold = record.get('gold', {})
    predictions = record.get('predictions', {})
    for question, expected, predicted in (
        ('loop_state', 'expected_loop_state', 'predicted_loop_state'),
        ('next_hand', 'expected_hand', 'predicted_hand'),
        ('high_risk_code', 'expected_high_risk_code', 'predicted_high_risk_code'),
    ):
        if question in gold:
            result[expected] = _label(gold[question])
            result[predicted] = _label(predictions.get(question))
        elif name == question and labels and 'label_index' in record:
            result[expected] = labels[int(record['label_index'])]
            probs = record.get('probabilities', [])
            index = max(range(len(probs)), key=probs.__getitem__) if probs else record.get('predicted_index')
            result[predicted] = labels[index] if index is not None else None
    if 'expected_loop_state' in result:
        result['expected_repeat'] = result['expected_loop_state'] == 'repeat_without_progress'
        result['predicted_repeat'] = result.get('predicted_loop_state') == 'repeat_without_progress'
    state = record.get('state', {})
    result['modality'] = record.get('modality', record.get('metadata', {}).get('modality',
                           state.get('modality') if isinstance(state, Mapping) else None))
    return result


def _classification(pairs, labels=None):
    labels = sorted(set(labels or []) | {str(x) for pair in pairs for x in pair})
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for expected, predicted in pairs:
        matrix[str(expected)][str(predicted)] += 1
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(row[label] for row in matrix.values())
        per_class[label] = {'support': support, 'predicted_positive': predicted,
                            'recall': tp / support if support else None,
                            'precision': tp / predicted if predicted else None,
                            'f1': 2 * tp / (support + predicted) if support + predicted else 0.0}
    return {'support': len(pairs), 'confusion_matrix': matrix, 'per_class': per_class,
            'macro_f1': sum(x['f1'] for x in per_class.values()) / len(labels) if labels else None}


def evaluate_acceptance(records, config=None):
    config = config or AcceptanceConfig()
    rows = [annotations(row) for row in records]
    gates = {}

    def gate(name, metrics, checks, support=None):
        enough = (metrics.get('support', 0) if support is None else support) >= config.min_support
        gates[name] = {'passed': enough and all(value is not None and
                       (value <= limit if direction == 'max' else value >= limit)
                       for value, limit, direction in checks.values()),
                       'metrics': metrics, 'checks': {key: {'value': value, 'threshold': limit,
                       'direction': direction} for key, (value, limit, direction) in checks.items()},
                       'sufficient_support': enough}

    precedence = evaluate_workflow_precedence()
    precedence['confusion_matrix'] = _classification([(c['expected'], c['actual']) for c in precedence['cases']])['confusion_matrix']
    gates['deterministic_guard_precedence'] = {'passed': precedence['accuracy'] == 1.0,
                                              'threshold': 1.0, 'metrics': precedence}
    def binary(name, subset, expected, predicted, recall, precision=None):
        def boolean(value):
            if value in (True, 'true'): return True
            if value in (False, 'false'): return False
            if value is None: return 'missing'
            raise ValueError(f'{name}: expected boolean label')
        pairs = [(boolean(r[expected]), boolean(r.get(predicted))) for r in subset if expected in r]
        metrics = _classification(pairs, ['False', 'True'])
        positive = metrics['per_class']['True']
        checks = {'recall': (positive['recall'], recall, 'min')}
        if precision is not None:
            checks['precision'] = (positive['precision'], precision, 'min')
        gate(name, metrics, checks, positive['support'])

    binary('repeat_risk', rows, 'expected_repeat', 'predicted_repeat', config.repeat_recall, config.repeat_precision)
    for modality in config.modalities:
        subset = [r for r in rows if r['modality'] == modality and 'expected_hand' in r]
        metrics = _classification([(r['expected_hand'], r.get('predicted_hand')) for r in subset])
        gate('next_hand_' + modality, metrics, {'macro_f1': (metrics['macro_f1'], config.next_hand_macro_f1, 'min')})
    binary('high_risk_code', [r for r in rows if r['modality'] == 'code'],
           'expected_high_risk_code', 'predicted_high_risk_code', config.code_recall, config.code_precision)
    pdf = [r for r in rows if r['modality'] == 'pdf' and r.get('expected_hand') in ('extract_pdf_text', 'render_pdf_page')]
    metrics = _classification([(r['expected_hand'], r.get('predicted_hand')) for r in pdf], ['extract_pdf_text', 'render_pdf_page'])
    gate('pdf_extract_vs_render', metrics, {'macro_f1': (metrics['macro_f1'], config.pdf_macro_f1, 'min')})
    images = [dict(r, expected_visual=r['expected_hand'] == 'inspect_image',
                   predicted_visual=r.get('predicted_hand') == 'inspect_image')
              for r in rows if r['modality'] == 'image' and 'expected_hand' in r]
    binary('image_visual_inspection', images, 'expected_visual', 'predicted_visual', config.image_recall)
    calibrated = [r for r in rows if 'probabilities' in r and 'label_index' in r]
    for r in calibrated:
        p = r['probabilities']
        if not p or any(not math.isfinite(x) or x < 0 for x in p) or not math.isclose(sum(p), 1, abs_tol=1e-5) or not 0 <= r['label_index'] < len(p):
            raise ValueError('invalid calibration distribution or label_index')
    metrics = dict(_summary(calibrated), support=len(calibrated))
    metrics['confusion_matrix'] = _classification([(r['label_index'], max(range(len(r['probabilities'])), key=r['probabilities'].__getitem__)) for r in calibrated])['confusion_matrix']
    gate('calibration', metrics, {'ece': (metrics['ece'], config.max_ece, 'max')})
    for field, limit in [('latency_ms', config.max_latency_p95_ms), ('vram_mb', config.max_vram_mb)]:
        values = sorted(float(r[field]) for r in rows if field in r)
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f'invalid {field}')
        value = (values[math.ceil(.95 * len(values)) - 1] if field == 'latency_ms' else max(values)) if values else None
        if limit is None:
            gates[field] = {'passed': None, 'status': 'not_required', 'metrics': {'support': len(values), 'value': value}}
        else:
            gate(field, {'support': len(values), 'value': value}, {'value': (value, limit, 'max')})
    return {'advisory_only': True, 'authorizes_execution': False,
            'note': 'The deterministic guard remains authoritative. Model output cannot authorize writes, retries, or stops.',
            'config': asdict(config), 'passed': all(g['passed'] for g in gates.values() if g['passed'] is not None), 'gates': gates}
