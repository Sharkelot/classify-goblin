"""Run against a separately started local server."""
import json
from classify_goblin import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient() as client:
    result = client.systemOne(
        state={'summary': 'Source code test result is partial'},
        questions={
            'route': Choice(instructions='Which evidence kind?', options={'code': 'source code', 'pdf': 'PDF text'}),
            'quality': Score(instructions='Evidence quality?', criteria=['missing', 'partial', 'complete']),
            'ready': Noul(instructions='Is evidence ready?', criteria={'true': 'complete', 'false': 'partial'}),
        },
    )
    print(json.dumps(result, indent=2))
