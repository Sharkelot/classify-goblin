"""Sanitized profile roster snapshot (CG-MM-11).

``ProfileEntry`` and ``RosterSnapshot`` are frozen, JSON-serializable.
``parse_profile_dir`` is the pure reader: it opens ONLY the allowlisted
files (``profile.yaml``, ``config.yaml``, the ``skills/`` directory) and
never touches ``.env``, ``auth.json``, ``memories/``, ``sessions/``, or any
other file in the profile directory. No bytes, raw reasoning, or secret
values ever enter an entry.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

# Files the parser is allowed to read, by basename.
ALLOWLISTED_FILES = ('profile.yaml', 'config.yaml')
ALLOWLISTED_DIRS = ('skills',)

# Keys we pull out of profile.yaml / config.yaml. Everything else is
# dropped so that secret-bearing keys (api_key, tokens, ...) never leak
# into the sanitized entry.
_PROFILE_KEYS = ('description', 'description_auto')
_MODEL_KEYS = ('default', 'provider', 'supports_vision', 'context_length')


@dataclass(frozen=True)
class ProfileEntry:
    name: str
    on_disk: bool
    valid_assignee: bool
    description: object
    description_auto: bool
    model_label: object
    provider_label: object
    context_length: object
    modalities: tuple
    skill_names: tuple
    capabilities: tuple
    workspace: object
    status: str
    snapshot_time: object
    status_reason: object = None

    def to_dict(self):
        return {
            'name': self.name,
            'on_disk': self.on_disk,
            'valid_assignee': self.valid_assignee,
            'description': self.description,
            'description_auto': self.description_auto,
            'model_label': self.model_label,
            'provider_label': self.provider_label,
            'context_length': self.context_length,
            'modalities': list(self.modalities),
            'skill_names': list(self.skill_names),
            'capabilities': list(self.capabilities),
            'workspace': self.workspace,
            'status': self.status,
            'status_reason': self.status_reason,
            'snapshot_time': self.snapshot_time,
        }

    def entry_digest(self):
        """Stable digest of the sanitized fields (never includes bytes)."""
        payload = json.dumps(self.to_dict(), sort_keys=True,
                            default=str).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RosterSnapshot:
    entries: tuple
    valid_assignees: tuple
    snapshot_time: object
    digest: object = None

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError('roster must be a JSON object')
        valid_assignees = data.get('valid_assignees')
        if not isinstance(valid_assignees, list):
            raise ValueError('valid_assignees must be a list')
        raw_entries = data.get('entries')
        if not isinstance(raw_entries, list):
            raise ValueError('entries must be a list')
        entries = []
        for raw in raw_entries:
            if not isinstance(raw, dict) or 'name' not in raw:
                raise ValueError('each entry must be an object with a name')
            modalities = raw.get('modalities')
            skills = raw.get('skill_names')
            caps = raw.get('capabilities')
            for label, val in (('modalities', modalities),
                              ('skill_names', skills),
                              ('capabilities', caps)):
                if val is not None and not isinstance(val, list):
                    raise ValueError(f'{label} must be a list')
            entries.append(ProfileEntry(
                name=raw['name'],
                on_disk=bool(raw.get('on_disk', True)),
                valid_assignee=bool(raw.get('valid_assignee', True)),
                description=raw.get('description'),
                description_auto=bool(raw.get('description_auto', False)),
                model_label=raw.get('model_label'),
                provider_label=raw.get('provider_label'),
                context_length=raw.get('context_length'),
                modalities=tuple(modalities or ()),
                skill_names=tuple(skills or ()),
                capabilities=tuple(caps or ()),
                workspace=raw.get('workspace'),
                status=raw.get('status', 'ok'),
                snapshot_time=raw.get('snapshot_time'),
                status_reason=raw.get('status_reason'),
            ))
        snapshot_time = data.get('snapshot_time')
        payload = json.dumps(
            {'entries': [e.to_dict() for e in entries],
             'valid_assignees': list(valid_assignees),
             'snapshot_time': snapshot_time},
            sort_keys=True, default=str).encode('utf-8')
        digest = data.get('digest') or hashlib.sha256(payload).hexdigest()
        return cls(entries=tuple(entries),
                   valid_assignees=tuple(valid_assignees),
                   snapshot_time=snapshot_time, digest=digest)

    def to_dict(self):
        return {
            'entries': [e.to_dict() for e in self.entries],
            'valid_assignees': list(self.valid_assignees),
            'snapshot_time': self.snapshot_time,
            'digest': self.digest,
        }


def _read_scalar_yaml(path):
    """Minimal flat key: value YAML reader (stdlib only).

    Supports the two-file shape Hermes profiles use: top-level scalar keys
    in profile.yaml, and a single nested ``model:`` block in config.yaml.
    Unknown keys are ignored (allowlisted extraction, not full parsing).
    Returns a dict of the extracted keys.
    """
    out = {}
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return out
    current = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if not line.startswith((' ', '\t')):
            current = None
            key, sep, value = line.partition(':')
            if not sep:
                continue
            key = key.strip()
            if key in _PROFILE_KEYS:
                out[key] = _yaml_scalar(value)
            elif key == 'model':
                current = 'model'
        elif current == 'model':
            key, sep, value = line.strip().partition(':')
            if not sep:
                continue
            key = key.strip()
            if key in _MODEL_KEYS:
                out[key] = _yaml_scalar(value)
    return out


def _yaml_scalar(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    if value in ('true', 'True'):
        return True
    if value in ('false', 'False'):
        return False
    if value in ('', '~', 'null', 'None'):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_profile_dir(path):
    """Build a sanitized ProfileEntry from a profile directory.

    Opens ONLY ``profile.yaml``, ``config.yaml`` (if present), and the
    ``skills/`` directory listing. Any other file is never opened, so
    ``.env``, ``auth.json``, ``memories/`` and ``sessions/`` cannot leak.
    A missing config.yaml yields status ``unavailable`` (with a reason),
    never an exception.
    """
    path = Path(path)
    name = path.name
    on_disk = path.is_dir()

    profile = _read_scalar_yaml(path / 'profile.yaml')
    config_path = path / 'config.yaml'
    config = _read_scalar_yaml(config_path) if config_path.is_file() else {}

    skills = ()
    skills_dir = path / 'skills'
    if skills_dir.is_dir():
        try:
            skills = tuple(sorted(p.name for p in skills_dir.iterdir()
                                  if p.is_dir()))
        except OSError:
            skills = ()

    modalities = ['text']
    if config.get('supports_vision') is True:
        modalities.append('vision')

    if config_path.is_file():
        status, reason = 'ok', None
    else:
        status, reason = 'unavailable', 'config_unavailable'

    return ProfileEntry(
        name=name,
        on_disk=on_disk,
        valid_assignee=True,  # the dispatcher revalidates; snapshot default
        description=profile.get('description'),
        description_auto=bool(profile.get('description_auto', False)),
        model_label=config.get('default'),
        provider_label=config.get('provider'),
        context_length=config.get('context_length'),
        modalities=tuple(modalities),
        skill_names=skills,
        capabilities=(),
        workspace=None,
        status=status,
        snapshot_time=None,
        status_reason=reason,
    )


def parse_profiles_root(root):
    """Parse every profile directory under ``root`` (sorted by name)."""
    root = Path(root)
    entries = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir():
                entries.append(parse_profile_dir(child))
    return tuple(entries)
