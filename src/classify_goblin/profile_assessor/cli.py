"""Profile assessor CLI (CG-MM-11).

Usage:
    python -m classify_goblin.profile_assessor.cli \
        --roster roster.json --request request.json
    python -m classify_goblin.profile_assessor.cli --profiles-root DIR

Exit codes:
    0  success (JSON result on stdout)
    2  missing/invalid input (file not found, bad JSON, schema violation)

The output is advisory-only JSON. No bytes, no secrets, no raw reasoning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support both `python -m classify_goblin.profile_assessor.cli` and
# `python src/classify_goblin/profile_assessor/cli.py`.
try:
    from .schema import AssessmentRequest
    from .roster import RosterSnapshot, parse_profiles_root
    from . import assess
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from classify_goblin.profile_assessor.schema import AssessmentRequest
    from classify_goblin.profile_assessor.roster import (
        RosterSnapshot, parse_profiles_root)
    from classify_goblin.profile_assessor import assess


def _load_json(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'file not found: {path}')
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='classify-goblin-profile-assessor',
        description='Advisory-only Hermes profile fit assessment.')
    parser.add_argument('--roster', help='path to roster JSON')
    parser.add_argument('--request', help='path to request JSON')
    parser.add_argument('--profiles-root',
                       help='parse profile dirs directly (no roster file)')
    args = parser.parse_args(argv)

    try:
        if args.profiles_root:
            entries = parse_profiles_root(args.profiles_root)
            raw = {
                'snapshot_time': None,
                'valid_assignees': [e.name for e in entries],
                'entries': [e.to_dict() for e in entries],
            }
            snapshot = RosterSnapshot.from_dict(raw)
            print(json.dumps({
                'entries': [e.to_dict() for e in entries],
                'advisory_only': True,
            }, sort_keys=True))
            return 0

        if not args.roster or not args.request:
            print('error: --roster and --request are required '
                  '(or use --profiles-root)', file=sys.stderr)
            return 2

        roster_data = _load_json(args.roster)
        request_data = _load_json(args.request)
        snapshot = RosterSnapshot.from_dict(roster_data)
        request = AssessmentRequest.from_dict(request_data)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2

    result = assess(request, snapshot)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
