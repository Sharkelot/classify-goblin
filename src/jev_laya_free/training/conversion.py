"""Offline, conservative observable-only conversion. No network dependencies."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .data import normalize_row, read_jsonl, split_examples, write_jsonl, _safe_local

VERSION = 1
LOOPS = ('progress', 'repeat_without_progress', 'blocked', 'terminal')
PRIVATE = re.compile(r'thought|reasoning|rationale|analysis|<think', re.I)


def observable(value):
    """Drop rationale fields and mixed free text rather than guess its boundaries."""
    if isinstance(value, dict):
        return {k: observable(v) for k, v in value.items() if not PRIVATE.search(k)}
    if isinstance(value, list):
        return [observable(v) for v in value]
    if isinstance(value, str) and PRIVATE.search(value):
        return '[mixed reasoning text omitted]'
    return _safe_local(value)


def decision(source, episode, step, state, name, labels, label):
    return normalize_row({
        'id': f'{episode}:{step}', 'group': episode, 'state': observable(state),
        'questions': {name: {'type': 'choice', 'instructions': f'Classify {name}.',
                              'criteria': dict.fromkeys(labels)}},
        'gold': {name: label},
    }, source=source, ordinal=step)


def hermes(rows):
    for i, row in enumerate(rows):
        gold = row.get('gold', {})
        label = gold.get('loop_state')
        if label not in LOOPS:
            continue
        meta = row.get('metadata', {})
        # Events from the same board task must remain in one split.
        task = meta.get('task_id_hash') or row.get('task_id') or str(row.get('id', '')).rsplit(':', 1)[0]
        if not task:
            continue
        episode = f"{meta.get('board', '')}:{task}"
        state = {k: v for k, v in row.get('state', {}).items() if k in {
            'task_status', 'priority', 'domain', 'observed_result', 'error_class',
            'retry_status', 'recurrences', 'consecutive_failures', 'run_status',
            'run_outcome', 'prior_event_kinds', 'trace_mode'}}
        yield decision('hermes', episode, i, state, 'loop_state', LOOPS, label)


def hazard(rows, partition):
    for row in rows:
        for k, outcome in enumerate(row['edit_outcomes']):
            if type(outcome) is not bool:
                raise ValueError('edit_outcomes must contain booleans')
            # Predict next observed edit outcome from a strictly preceding prefix.
            ex = decision('agenthazard', row['id'], k,
                          {'scaffold': row['scaffold'], 'edit_outcomes': row['edit_outcomes'][:k]},
                          'next_edit_outcome', ('clean', 'error'), 'error' if outcome else 'clean')
            ex.metadata['official_split'] = partition
            yield ex


def eto(rows, domain):
    for row in rows:
        episode = f"{domain}:{row.get('game_file', row['id'])}"
        observation = None
        previous_action = None
        previous_observation = None
        for k, turn in enumerate(row['conversations'][2:], 2):
            if turn['from'] == 'human':
                observation = turn['value']
                continue
            # Only the single Action line is observable. Never retain assistant prose.
            match = re.search(r'^Action:\s*([^\r\n]+)$', turn['value'], re.M)
            if turn['from'] != 'gpt' or not match or observation is None:
                continue
            action = match.group(1).strip()
            if PRIVATE.search(action):
                continue
            repeated = action == previous_action and observation == previous_observation
            yield decision('eto', episode, k, {'observation': observation, 'action': action},
                           'repeated_action_observation', ('false', 'true'), str(repeated).lower())
            previous_action, previous_observation = action, observation


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build(root, hermes_path, out, config, *, local=False):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    partitions = {'train': [], 'validation': [], 'test': []}
    report = {'converter_version': VERSION, 'mode': 'local-only' if local else 'public',
              'config_sha256': sha(Path(config)), 'sources': []}
    for spec in json.loads(Path(config).read_text())['sources']:
        entry = dict(spec, files=[], examples=0)
        report['sources'].append(entry)
        paths = [Path(hermes_path)] if spec['adapter'] == 'hermes' else sorted(root.glob(spec['glob']))
        paths = [p for p in paths if p.is_file()]
        if spec['adapter'] == 'hazard':
            paths = [p for p in paths if p.name in {'train.json', 'val.json', 'test.json'}]
        entry['available_files'] = len(paths)
        if not paths:
            entry['status'] = 'unavailable'
            continue
        if spec['adapter'] == 'manifest-only':
            entry['status'] = 'excluded'
            continue
        if spec['policy'] == 'local-only' and not local:
            entry['status'] = 'excluded-local-only'
            continue
        card = root / spec.get('license_file', '')
        if spec['policy'] == 'public':
            if spec['license'] not in {'Apache-2.0', 'CC-BY-4.0'} or not card.is_file() or spec['license_marker'] not in card.read_text():
                entry['status'] = 'excluded-license-unverified'
                continue
            entry['license_evidence_sha256'] = sha(card)
        entry['status'] = 'converted'
        examples = []
        for path in paths:
            entry['files'].append({'path': path.name if spec['adapter'] == 'hermes' else str(path.relative_to(root)), 'sha256': sha(path)})
            if spec['adapter'] == 'hermes':
                converted = list(hermes(read_jsonl(path)))
            elif spec['adapter'] == 'hazard':
                partition = {'train': 'train', 'val': 'validation', 'test': 'test'}[path.stem]
                converted = list(hazard(json.loads(path.read_text()), partition))
            elif spec['adapter'] == 'eto':
                converted = list(eto(json.loads(path.read_text()), path.stem))
            else:
                raise ValueError('unknown adapter')
            for ex in converted:
                ex.metadata.update({'license': spec['license'], 'artifact_policy': spec['policy'],
                                    'input_sha256': entry['files'][-1]['sha256'], 'converter_version': VERSION})
            examples.extend(converted)
        entry['examples'] = len(examples)
        entry['episodes'] = len({e.group for e in examples})
        entry['label_coverage'] = dict(Counter(f'{name}:{list(e.questions[name]["criteria"])[target["label_index"]]}' for e in examples for name, target in e.targets.items()))
        for e in examples:
            if 'official_split' in e.metadata:
                partitions[e.metadata['official_split']].append(e)
        train, validation = split_examples(e for e in examples if 'official_split' not in e.metadata)
        partitions['train'].extend(train)
        partitions['validation'].extend(validation)
    report['splits'] = {}
    seen = set()
    for name, examples in partitions.items():
        groups = {e.group for e in examples}
        if seen & groups:
            raise ValueError('episode leakage across partitions')
        seen |= groups
        path = out / f'{name}.jsonl'
        count = write_jsonl(path, examples)
        report['splits'][name] = {'examples': count, 'episodes': len(groups), 'sha256': sha(path)}
    (out / 'manifest.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-root', type=Path, required=True)
    parser.add_argument('--hermes-path', type=Path, default=Path('/home/coreys/models/laya-sidecar/data/hermes-traces.jsonl'))
    parser.add_argument('--config', type=Path, default=Path('configs/dataset_sources.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--include-local', action='store_true')
    args = parser.parse_args()
    report = build(args.external_root, args.hermes_path, args.out, args.config, local=args.include_local)
    print(json.dumps({'splits': report['splits'], 'sources': {s['id']: s['status'] for s in report['sources']}}, indent=2))


if __name__ == '__main__':
    main()
