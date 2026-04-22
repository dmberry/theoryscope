"""
Translated Corpus Probe.

Re-run eigendecomposition on a machine-translated version of the
corpus and align the two bases via document projections (the same
basis-dim-independent primitive used by the Embedding Dependence
Probe, because translation may change the embedding dimension when
a multilingual model is used; for Marian + sentence-transformers,
the dimension usually stays the same but the content shifts).

Translation is done locally via HuggingFace Marian models
(Helsinki-NLP/opus-mt-en-*). First run per language downloads a
model (~300 MB). Results are cached on disk so repeated probes
are cheap.

Methodological claim: eigendirections that survive translation
arguably track concepts; directions that do not arguably track
language-specific framings. The translation is not neutral, and the
delta is what matters.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
from sklearn.decomposition import PCA

from corpus.embed import embed_texts
from corpus.pipeline import CorpusSpec, build_provenance, ingest_and_embed
from corpus.translations import load_translations, save_translations
from geometry.eigen_align import align_via_doc_projection

logger = logging.getLogger(__name__)


# Short-list of open-weight Marian translation models from Helsinki-NLP.
# Every first-run downloads ~300 MB per language.
AVAILABLE_LANGUAGES: List[Dict[str, str]] = [
    {
        "code": "fr",
        "label": "French (Marian)",
        "model_id": "Helsinki-NLP/opus-mt-en-fr",
    },
    {
        "code": "de",
        "label": "German (Marian)",
        "model_id": "Helsinki-NLP/opus-mt-en-de",
    },
    {
        "code": "es",
        "label": "Spanish (Marian)",
        "model_id": "Helsinki-NLP/opus-mt-en-es",
    },
    {
        "code": "zh",
        "label": "Chinese (Marian)",
        "model_id": "Helsinki-NLP/opus-mt-en-zh",
    },
]


def list_available_languages() -> List[Dict[str, str]]:
    return list(AVAILABLE_LANGUAGES)


def _resolve_language(target_lang: str) -> Dict[str, str]:
    for entry in AVAILABLE_LANGUAGES:
        if entry["code"] == target_lang or entry["model_id"] == target_lang:
            return entry
    raise ValueError(
        f"Unknown target language '{target_lang}'. Supported: "
        + ", ".join(e["code"] for e in AVAILABLE_LANGUAGES)
    )


def _translate_documents(
    texts: List[str],
    model_id: str,
    batch_size: int = 8,
) -> List[str]:
    """Translate a list of document texts with a Marian model.

    Lazy-imported so the backend can be inspected without the
    heavyweight `transformers` import path firing. Each batch runs a
    generate pass on the Marian model.
    """
    try:
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "The `transformers` package is required for translation. "
            "Run backend/setup.sh to install the backend requirements."
        ) from e

    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id)
    model.eval()

    outputs: List[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        translated_ids = model.generate(**encoded, max_new_tokens=512)
        decoded = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
        outputs.extend(decoded)
    return outputs


def compute_translation_probe(
    spec: CorpusSpec,
    target_lang: str,
    n_components: int = 5,
    n_samples: int = 3,
) -> Dict[str, Any]:
    language = _resolve_language(target_lang)
    bundle = ingest_and_embed(spec)
    n_docs = bundle.embeddings.shape[0]
    if n_docs < 4:
        raise ValueError(
            f"Corpus has only {n_docs} document(s); Translated Corpus Probe needs at least 4."
        )
    n_components = max(2, min(int(n_components), min(n_docs, 20)))

    # Corpus hash drives the translation cache key.
    corpus_hash = hashlib_hex(bundle)

    cached = load_translations(
        corpus_hash=corpus_hash,
        target_lang=language["code"],
        model_id=language["model_id"],
    )
    if cached is None:
        logger.info(
            "Translating %d documents to %s via %s",
            n_docs,
            language["code"],
            language["model_id"],
        )
        translated_texts = _translate_documents(
            [d.text for d in bundle.documents],
            model_id=language["model_id"],
        )
        translations = {
            d.id: translated_texts[i] for i, d in enumerate(bundle.documents)
        }
        save_translations(
            corpus_hash=corpus_hash,
            target_lang=language["code"],
            model_id=language["model_id"],
            translations=translations,
        )
        cache_hit = False
    else:
        translations = cached
        cache_hit = True

    # Re-embed the translated corpus with the baseline embedding model.
    translated_in_order = [translations.get(d.id, d.text) for d in bundle.documents]
    translated_embeddings = embed_texts(
        translated_in_order, model_id=bundle.embedding_spec.model_id
    )

    # PCA on both the original and the translated corpus.
    pca_baseline = PCA(n_components=n_components)
    coords_baseline = pca_baseline.fit_transform(bundle.embeddings).astype(
        np.float32
    )
    pca_translated = PCA(n_components=n_components)
    coords_translated = pca_translated.fit_transform(translated_embeddings).astype(
        np.float32
    )

    alignment = align_via_doc_projection(coords_baseline, coords_translated)

    # Carry a few sample translations for the UI to show alongside originals.
    sample_indices = list(range(min(int(n_samples), n_docs)))
    samples = [
        {
            "id": bundle.documents[i].id,
            "author": bundle.documents[i].author,
            "year": bundle.documents[i].year,
            "title": bundle.documents[i].title,
            "original_text": bundle.documents[i].text,
            "translated_text": translations.get(
                bundle.documents[i].id, ""
            ),
        }
        for i in sample_indices
    ]

    provenance = build_provenance(
        bundle=bundle,
        operation="translation_probe",
        operator_name="marian_then_align",
        operator_params={
            "target_lang": language["code"],
            "translation_model_id": language["model_id"],
            "n_components": n_components,
        },
    )

    return {
        "target_lang": language["code"],
        "language_label": language["label"],
        "translation_model_id": language["model_id"],
        "cache_hit": cache_hit,
        "baseline_variance_explained": [
            float(v) for v in pca_baseline.explained_variance_ratio_
        ],
        "translated_variance_explained": [
            float(v) for v in pca_translated.explained_variance_ratio_
        ],
        "alignment": {
            "matches": [asdict(m) for m in alignment.matches],
            "per_component": alignment.per_component,
            "stability": alignment.stability,
        },
        "samples": samples,
        "provenance": provenance.to_dict(),
    }


def hashlib_hex(bundle) -> str:
    """Stable hash over the bundle's document IDs for cache keying."""
    import hashlib

    ids = sorted(d.id for d in bundle.documents)
    blob = "|".join(ids).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]
