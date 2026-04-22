"""
On-disk translation cache.

Translations are expensive (each call loads a Marian model, runs
inference, and produces a new text). We cache per-document
translations keyed by (doc_id, target_lang, model_id) so repeated
calls against the same document are free.

Cached files live under .theoryscope-cache/translations/ alongside
the corpus cache. One JSON file per (corpus_hash, target_lang).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from .cache import TRANSLATIONS_DIR, ensure_cache_dirs


def _translation_key(corpus_hash: str, target_lang: str, model_id: str) -> str:
    blob = f"{corpus_hash}:{target_lang}:{model_id}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def _translation_path(key: str) -> Path:
    return TRANSLATIONS_DIR / f"{key}.json"


def load_translations(
    corpus_hash: str,
    target_lang: str,
    model_id: str,
) -> Optional[Dict[str, str]]:
    """Return {doc_id: translated_text} or None on cache miss."""
    key = _translation_key(corpus_hash, target_lang, model_id)
    path = _translation_path(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("translations"), dict):
            return payload["translations"]
    except Exception:
        return None
    return None


def save_translations(
    corpus_hash: str,
    target_lang: str,
    model_id: str,
    translations: Dict[str, str],
) -> None:
    ensure_cache_dirs()
    key = _translation_key(corpus_hash, target_lang, model_id)
    path = _translation_path(key)
    payload = {
        "corpus_hash": corpus_hash,
        "target_lang": target_lang,
        "model_id": model_id,
        "translations": translations,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def list_cached(corpus_hash: str) -> List[Dict[str, str]]:
    """List cached translations for a corpus, if the directory exists."""
    ensure_cache_dirs()
    out: List[Dict[str, str]] = []
    for f in TRANSLATIONS_DIR.glob("*.json"):
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
            if meta.get("corpus_hash") == corpus_hash:
                out.append(
                    {
                        "target_lang": meta.get("target_lang", ""),
                        "model_id": meta.get("model_id", ""),
                    }
                )
        except Exception:
            continue
    return out
