"""Local artifact contract. Resolved bytes are internal and must never enter usage.

PDF selections are one-based; crops are [x0, y0, x1, y1] in unit coordinates.
A PDF-aware backend must validate selected pages against the decoded document.
"""
import errno
import hashlib
import os
import re
import stat
from dataclasses import dataclass, field
from .schema import require, ValidationError

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MIMES = {'image': {'image/png', 'image/jpeg', 'image/webp'},
         'pdf': {'application/pdf'}, 'text': {'text/plain'},
         'code': {'text/plain', 'text/x-python', 'application/javascript',
                  'text/javascript', 'application/json'}}


class ArtifactSelectionError(ValueError):
    """Adapters raise this when a selection is invalid for the decoded artifact."""


class ArtifactUnavailable(RuntimeError):
    """Safe service failure; never expose underlying filesystem exceptions."""


def validate(items):
    require(isinstance(items, list) and len(items) <= 8, 'artifacts must be a list of at most 8 references')
    result, ids = [], set()
    for a in items:
        require(isinstance(a, dict), 'artifact must be an object')
        require(set(a) <= {'id', 'kind', 'mime', 'sha256', 'path', 'pages', 'crops'}
                and {'id', 'kind', 'mime', 'sha256', 'path'} <= set(a), 'invalid artifact fields')
        ident = a['id']
        require(isinstance(ident, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', ident) is not None,
                'invalid artifact id')
        require(ident not in ids, 'duplicate artifact id')
        ids.add(ident)
        kind, mime = a['kind'], a['mime']
        require(isinstance(kind, str) and kind in MIMES, 'unsupported artifact kind')
        require(isinstance(mime, str) and mime in MIMES[kind], 'unsupported artifact MIME')
        require(isinstance(a['sha256'], str) and re.fullmatch(r'[0-9a-fA-F]{64}', a['sha256']) is not None,
                'invalid artifact sha256')
        path = a['path']
        require(isinstance(path, str) and 0 < len(path) <= 1024 and
                not any(ord(c) < 32 for c in path) and not any(c in path for c in '\\:') and
                all(p not in ('', '.', '..') for p in path.split('/')), 'artifact path must be relative without traversal')
        pages = a.get('pages', [1])
        require(isinstance(pages, list) and 1 <= len(pages) <= 4 and
                all(type(p) is int and 1 <= p <= 2147483647 for p in pages), 'invalid artifact pages (maximum 4)')
        require(len(set(pages)) == len(pages), 'duplicate artifact page')
        require(kind == 'pdf' or 'pages' not in a, 'pages require a PDF artifact')
        crops = a.get('crops', {})
        require(isinstance(crops, dict), 'artifact crops must be a page mapping')
        for page, boxes in crops.items():
            require(page in [str(p) for p in pages], 'crop page must be selected')
            require(kind in ('pdf', 'image'), 'crops require image or PDF')
            require(isinstance(boxes, list) and 1 <= len(boxes) <= 4, 'maximum 4 crops per page')
            for box in boxes:
                require(isinstance(box, list) and len(box) == 4 and
                        all(type(v) in (int, float) and 0 <= v <= 1 for v in box), 'invalid normalized crop')
                require(box[0] < box[2] and box[1] < box[3], 'crop must have positive area')
        result.append({**a, 'sha256': a['sha256'].lower()})
    return result


@dataclass(frozen=True)
class ResolvedArtifact:
    id: str
    kind: str
    mime: str
    sha256: str
    pages: tuple
    crops: dict
    data: bytes = field(repr=False)

    def usage(self):
        return {'id': self.id, 'pages': list(self.pages), 'truncated': False}


class ArtifactBroker:
    def __init__(self, roots=None):
        self.roots = tuple(roots if roots is not None else filter(None, os.environ.get('JEV_ARTIFACT_ROOTS', '').split(os.pathsep)))
        if any(not os.path.isabs(root) for root in self.roots):
            raise ArtifactUnavailable('artifact roots must be absolute')

    def resolve(self, items):
        return [self._resolve(a) for a in validate(items)]

    def _resolve(self, a):
        if not self.roots:
            raise ArtifactUnavailable('artifact broker unavailable')
        for root in self.roots:
            fd = None
            try:
                fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                parts = a['path'].split('/')
                for index, part in enumerate(parts):
                    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                    if index < len(parts) - 1:
                        flags |= os.O_DIRECTORY
                    child = os.open(part, flags, dir_fd=fd)
                    os.close(fd)
                    fd = child
                info = os.fstat(fd)
                require(stat.S_ISREG(info.st_mode), 'artifact must be a regular file')
                require(info.st_size <= MAX_ARTIFACT_BYTES, 'artifact exceeds 64 MiB')
                with os.fdopen(fd, 'rb') as stream:
                    fd = None
                    data = stream.read(MAX_ARTIFACT_BYTES + 1)
                require(len(data) <= MAX_ARTIFACT_BYTES, 'artifact exceeds 64 MiB')
                require(hashlib.sha256(data).hexdigest() == a['sha256'], 'artifact digest mismatch')
                return ResolvedArtifact(a['id'], a['kind'], a['mime'], a['sha256'],
                                        tuple(a.get('pages', [1]) if a['kind'] == 'pdf' else []),
                                        a.get('crops', {}), data)
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    continue
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise ValidationError('artifact symlinks or invalid path components rejected') from None
                raise ArtifactUnavailable('artifact broker unavailable') from None
            finally:
                if fd is not None:
                    os.close(fd)
        raise ValidationError('artifact reference not found in configured roots')
