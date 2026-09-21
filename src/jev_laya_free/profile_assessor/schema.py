"""Request schema for the profile assessor (JEV-MM-11).

``AssessmentRequest`` is a frozen, JSON-serializable description of the work
being routed. ``from_dict`` validates types and enum values and raises
``ValueError`` on anything malformed so the CLI can fail closed (exit 2)
rather than guess.
"""
from __future__ import annotations

from dataclasses import dataclass

_RISK = ('low', 'medium', 'high')
_URGENCY = ('normal', 'high')
_COST = ('low', 'medium', 'high')


@dataclass(frozen=True)
class AssessmentRequest:
    task_kind: str
    required_modalities: tuple = ()
    min_context: int = 0
    required_capabilities: tuple = ()
    workspace_repo: object = None
    risk: str = 'low'
    urgency: str = 'normal'
    cost_hint: str = 'low'

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError('request must be a JSON object')
        task_kind = data.get('task_kind')
        if not isinstance(task_kind, str) or not task_kind:
            raise ValueError('task_kind must be a non-empty string')

        def _str_list(key, default=()):
            val = data.get(key, default)
            if val is None:
                return default
            if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
                raise ValueError(f'{key} must be a list of strings')
            return tuple(val)

        min_context = data.get('min_context', 0)
        if not isinstance(min_context, int) or isinstance(min_context, bool):
            raise ValueError('min_context must be an integer')

        workspace_repo = data.get('workspace_repo', None)
        if workspace_repo is not None and not isinstance(workspace_repo, str):
            raise ValueError('workspace_repo must be a string or null')

        risk = data.get('risk', 'low')
        if risk not in _RISK:
            raise ValueError(f'risk must be one of {_RISK}')
        urgency = data.get('urgency', 'normal')
        if urgency not in _URGENCY:
            raise ValueError(f'urgency must be one of {_URGENCY}')
        cost_hint = data.get('cost_hint', 'low')
        if cost_hint not in _COST:
            raise ValueError(f'cost_hint must be one of {_COST}')

        return cls(
            task_kind=task_kind,
            required_modalities=_str_list('required_modalities'),
            min_context=min_context,
            required_capabilities=_str_list('required_capabilities'),
            workspace_repo=workspace_repo,
            risk=risk,
            urgency=urgency,
            cost_hint=cost_hint,
        )

    def to_dict(self):
        return {
            'task_kind': self.task_kind,
            'required_modalities': list(self.required_modalities),
            'min_context': self.min_context,
            'required_capabilities': list(self.required_capabilities),
            'workspace_repo': self.workspace_repo,
            'risk': self.risk,
            'urgency': self.urgency,
            'cost_hint': self.cost_hint,
        }
