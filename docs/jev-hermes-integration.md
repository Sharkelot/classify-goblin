# JEV typed advisory layer for Hermes

The loopback endpoint is `POST http://127.0.0.1:8093/v1/systemone`. Requests contain `model`, a JSON `state`, and the typed `questions` returned by `jev_laya_free.taxonomy.questions()`. The catalog covers model routing, confidence-gated actions, tool screening, progress, completion, skill selection, compaction, citation verification, RAG filtering, semantic find, composite scoring, and intent routing.

`jev_laya_free.workflow.decide(state, client)` sends only bounded routing metadata and returns `advisory.authoritative=false`. The deterministic guard and route are authoritative and cannot be overridden by probabilities.

| capability | Hermes hook | type | code policy |
|---|---|---|---|
| model routing | per-turn model picker | Choice | max probability >= .70, otherwise human |
| confidence action | before reversible/risky action | Score + Noul | .60 read, .90 writes, otherwise human |
| tool screening | pre-tool guard | Score | >= .20 review, >= .70 block |
| progress | trajectory gate | Choice | deterministic guard decides continue/warn/replan/halt |
| completion | stop/completion hook | Noul | review on positive; outage fails open |
| skill selection | skill loader | Score | read top 3 only at >= .65 |
| compaction | context pruner | Score | drop only below .35 |
| citation | citation checker | Noul | human review below .80 |
| RAG filtering | retrieval filter | Choice + Noul | injection always review; keep >= .65 |
| semantic find | candidate reranker | Score + Noul | answer-exists gates empty result |
| composite | ranking/scoring code | Score | weights remain code-owned |
| intent routing | handoff selector | Choice | max probability >= .70, otherwise human |

All backend errors fail open to the deterministic decision. No question authorizes writes, retries, tool calls, routing, or completion. The model is advisory and probabilities require local held-out calibration; the deterministic guard must remain 100% authoritative. The service is loopback-only and uses no hosted API.
