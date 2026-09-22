"""No service required. Qwen remains responsible for executing the route."""
import json
from classify_goblin.workflow import decide

state = {
    'modality': 'pdf', 'source_digest': 'synthetic-document-hash', 'location': 'page 2',
    'extraction_quality': 'scanned', 'action': 'extract page 2 text',
    'result': 'empty text', 'same_action_streak': 1,
}
print(json.dumps(decide(state), indent=2))
# To obtain advisory answers, pass a configured local TypeSafeClient as client=.
# Honor gate.decision first. Neither this adapter nor Laya executes the next hand.
