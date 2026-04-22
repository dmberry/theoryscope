> Part of the [Vector Lab](https://github.com/vector-lab-tools) — research instruments for critical vector theory.
> [Website and map](https://vector-lab-tools.github.io) · [Org profile](https://github.com/vector-lab-tools)
>
> **Tier:** single-object scope. **Object:** a corpus of theoretical texts.
>
> **Sibling instruments:** [Vectorscope](https://github.com/vector-lab-tools/vectorscope) · [Manifoldscope](https://github.com/vector-lab-tools/manifoldscope) · [Manifold Atlas](https://github.com/vector-lab-tools/manifold-atlas) · [LLMbench](https://github.com/vector-lab-tools/LLMbench)

# Theoryscope

**A renormalisation-group workbench for corpora of theoretical texts.**

**Author:** David M. Berry
**Institution:** University of Sussex
**Version:** 0.6.0
**Date:** 17 April 2026
**Licence:** MIT

Theoryscope is a web-based research tool that treats a corpus of theoretical texts as a geometry and asks renormalisation-group and eigenvector questions about it. It ingests a corpus, embeds each text with an open-weight model, and provides operations for inspecting the cloud's structure using vocabulary drawn from renormalisation group theory and linear algebra: principal eigendirections of a field, coarse-graining flows, fixed points, relevant and irrelevant operators, and universality classes.

The tool is designed for critical inquiry into the geometry of intellectual fields, not bibliometric measurement. Where existing computational literature tools (citation graph analyses, topic models, bibliometric dashboards) count and cluster surface features, Theoryscope provides an instrument for making visible the gap between what a field says it is about and what its geometry suggests it is about.

## Scholarly Context

Theoryscope emerges from the convergence of three research programmes.

**The Vector Lab.** The tool is the fourth instrument in a family alongside Manifold Atlas (between-model comparison of output embeddings), Vectorscope (within-model inspection of open-weight internals), and LLMbench (close reading of model outputs). Manifold Atlas and Vectorscope take a model as their object; Theoryscope takes a *corpus* as its object, extending the Vector Lab's critical posture beyond the machine to the thought that is written about the machine. All four instruments share the editorial design system, an open-weight-only methodology, and a commitment to making every parameter visible on screen rather than hidden behind opaque kernels.

**The renormalisation group as critical method.** Berry (2026) *Renormalising Theory* argues that claims critics already perform intuitively, that "Foucault, Deleuze and Guattari really come down to X", or that "Habermas and Rorty converge at depth", are in fact renormalisation-group claims about what survives when a field is coarse-grained. The physics vocabulary of flow on theory space (Fisher 1974), relevant and irrelevant operators, universality classes, and the mapping to deep learning (Mehta and Schwab 2014; Lin, Tegmark and Rolnick 2017) provides a formal language for a critical move that has always happened through the reading. Theoryscope is the instrument that follows from the argument.

**Counterfactual media and the theory-space concept.** Berry (2026) *What Is Theory Space?* treats theory space as the inaccessible totality of counterfactual media from which a particular manifold is selected during training. That post is about what cannot be inspected. Theoryscope works on a cognate but distinct object, a corpus of theoretical texts that is empirically given. The kinship is methodological. The difference is that Theoryscope's corpus space is available for inspection, coarse-graining, and re-embedding. It is the empirical companion to the conceptual argument.

## Corpora and Embeddings: A Primer

Theoryscope uses two kinds of object, and it helps to understand the difference.

A **corpus** is a finite, explicitly defined set of documents: a Zotero collection, an author's complete works, a time-bounded selection of a field. A corpus is a theoretical claim in its own right. Who is in it and who is out, under what criterion, with what filters, determines what the instrument can see. Theoryscope records the corpus definition as part of every result so that findings can be reproduced and contested.

An **embedding model** (MiniLM, BGE, Nomic, Qwen3-Embedding) takes text in and produces a **vector** out: a list of numbers (typically 384 to 1024 floating-point values). No language comes back, just coordinates in a high-dimensional space. Those coordinates are the model's geometric encoding of what the text "means", where meaning is reduced to position. The embedding model is the telescope: it converts text into a location.

This matters for Theoryscope because every operation is sensitive to which embedding model is computing the distances. The same corpus, embedded by two different open-weight models, will not produce the same eigendirections. The Embedding Dependence Probe exists to make this visible. An operation that reports a single eigenbasis without its stability under re-embedding is reporting half a result. Theoryscope uses only open-weight embedding models, whose architectures, training data, and update schedules are available for scholarly inspection.

The difference between a **chat model** and an **embedding model** is relevant here too. A chat model (GPT, Claude, Llama) takes text in and produces text out. An embedding model stops earlier: it produces the vector and hands it to you. This is why embedding API calls are cheap (fractions of a penny) while chat API calls are expensive. Theoryscope uses embedding models as its core instrument. Chat models appear only in later phases for the semantic coarse-graining operator, where an LLM is used to produce progressively more abstract paraphrases of a text before re-embedding.

## Operations at a Glance

Theoryscope is organised as a tabbed workspace with five groups: three operational groups (Inspect, Flow, Critique) and two cross-cutting features (Annotations, Atlas). Live at v0.6.0: Inspect complete (5 ops), Flow complete (5 ops), Critique backends complete (6 ops — three shipped with frontend, three backend-only pending UI). `GET /status` lists all 16 operations.

| Group | Operation | Status | Core question |
|---|---|---|---|
| **Inspect** | Corpus Map | ✅ live | What does the cloud look like? |
| Inspect | Eigendirections | ✅ live | What are the principal axes of the field? |
| Inspect | Debated vs Computed | ✅ live | Does the axis the field says it debates match a computed PC? |
| Inspect | Concept Locator | ✅ live | Where does this concept sit, and along which axis does it vary? |
| Inspect | Author Constellation | ✅ live | How do authors differ and where do they concentrate? |
| **Flow** | Coarse-Graining Trajectory | ✅ live | How do points move under successive coarse-grainings? |
| Flow | Fixed Point Finder | ✅ live | Where does the flow stop? |
| Flow | Universality Class Finder | ✅ live | Which starting positions converge to the same endpoint? |
| Flow | Relevant / Irrelevant Operator Spectrum | ✅ live | Which concepts persist and which wash out? |
| Flow | Temporal RG Flow | ✅ live | How do the principal axes shift across decades? |
| **Critique** | Embedding Dependence Probe | ✅ live | Do these results survive a different embedding? |
| Critique | Perturbation Test | ✅ live | How robust is this result to a small change in the corpus? |
| Critique | Forgetting Curve | ✅ live | How fragile are these eigendirections under bootstrap? |
| Critique | Symmetry Breaking Map | 🧪 backend | Where does the field fragment, and along which axis? |
| Critique | Phase Diagram | 🧪 backend | What does the flow field look like as a whole? |
| Critique | Translated Corpus Probe | 🧪 backend | What survives translation? |
| Critique | Corpus-vs-Model Probe | ⏳ planned | Do the eigendirections of a corpus match those of a model trained on it? |
| **Annotations** | Critical Annotations Layer | ⏳ planned | Markdown notes attached to any geometric feature, exported with the result. |
| **Atlas** | Curated pre-run analyses | ⏳ planned | Curated entries on named corpora, with provenance and annotations. |

All live operations carry a provenance record (corpus hash, embedding model, chunking spec, coarse-graining operator, optional stability score) that travels with every export. Results from the three Critique operations are recorded in the same format so that any downstream claim about the corpus can be cross-checked for robustness.

## Inspect Mode

The opening workspace. Load a corpus, watch it appear as geometry, read its principal axes.

### Core features

- **Open-weight embeddings.** Following the commitment to make every parameter inspectable, Theoryscope does not use commercial embedding APIs. The default embedder is `sentence-transformers/all-MiniLM-L6-v2`; the Embedding Dependence Probe uses `all-mpnet-base-v2` and `BAAI/bge-small-en-v1.5` as alternatives. All embedding models run locally via `sentence-transformers`.
- **Corpus as a cloud.** ✅ Each document is a point in a high-dimensional embedding space. The Corpus Map projects to three dimensions via PCA and renders the cloud as a 3D scatter in Plotly, with documents coloured by year and labelled by author.
- **Annotated eigendirections.** ✅ Each principal component is annotated with the documents that load most positively and most negatively on it. The critical use is comparison with the field's debated oppositions: computed axes and debated axes rarely match, and the gap is diagnostic.
- **Concept locator.** ✅ Enter a short text or concept. The tool embeds it in the same space, returns the nearest documents and authors, and reports the principal component most aligned with the query, with per-component alignment bars.
- **Author constellations.** ✅ Aggregate each author's documents to a centroid with intra-author spread. Overlaid on the same PCA basis as the Corpus Map. Click an author card or a constellation on the map to focus; an author-pair cosine matrix sits in the deep dive.
- **Debated vs computed.** ✅ Name an opposition the field is said to be structured by (with a short text per pole). The tool embeds the poles, constructs a direction vector, and reports the principal component most aligned with it, with a dominance score combining alignment and variance.
- **Provenance record.** ✅ Every operation view has a deep-dive panel exposing the full provenance JSON: corpus source, document IDs, embedding model and dimension, chunking spec, ingestion timestamp, and SHA-256 corpus hash. The `ExportButton` downloads the full operation result with its provenance as a timestamped JSON file.

## Flow Modes

The second tab group. Coarse-grain the corpus, watch it flow, locate its fixed points.

### Coarse-Graining Trajectory ✅
Applies a coarse-graining operator in stepwise passes, animating how points move. The operator is visible on screen with its parameters at every step. v0.3.0 implements the aggregative operator (progressive k-means with a log-spaced schedule from the full corpus down to two clusters); the semantic, lexical, and citation-graph operators are planned. A Play/Pause scrubber and clickable schedule bar let the user step through the flow manually.

### Fixed Point Finder ✅
Iterates the flow to its terminal step and reports the basins every document falls into. Basins are the fixed points of the flow: positions that do not move under further coarse-graining within the terminal cluster count. The terminal map is rendered with basins colour-coded; clicking a basin card highlights its members on the map. Each basin lists its population, exemplar document (the member closest to the basin centroid), and full member list.

### Relevant / Irrelevant Operator Spectrum
For a user-supplied or auto-extracted list of concepts, ranks each by whether its role in discriminating texts grows or shrinks under coarse-graining. Relevant operators persist at every scale; irrelevant operators wash out. The critical interpretation is the reader's; the tool reports the pattern.

### Relevant / Irrelevant Operator Spectrum ✅
User-supplied concepts are embedded in the corpus space, and their discriminating power across documents is tracked under progressive k-means coarse-graining. The relevance score per concept is the mean variance-ratio across every step of the schedule. Concepts whose axis of variation aligns with what the flow preserves score near 1.0 (structural operators); concepts whose variance the flow averages out score near 0.0 (surface operators). Ships with six default concepts (technology, labour, power, phenomenology, democracy, capital); the user can add, edit, and remove freely.

### Temporal RG Flow ✅
Coarse-grain by progressively wider year windows rather than by k-means. Each step bins documents into windows of a given width (from 1 year up through the full corpus span), places every document at its bin's centroid, and projects onto the shared PCA-2D basis. The temporal flow is a different critical question from semantic abstraction: it asks whether a field's variation is historical or conceptual. Play / Pause scrubber and clickable schedule of widths; bin listings in the deep dive.

### Universality Class Finder ✅
Clusters corpus points by the fixed point they flow to, rather than by surface embedding. Classes are ranked by ascending surface mean cosine: a class whose members were surface-different but converge on the same basin is the universality-class finding worth inspecting. Each class card shows a universality bar (1 − surface cosine) so the relative universality of classes is legible at a glance.

### Temporal RG Flow
Coarse-grains the corpus by time window rather than by semantic or aggregative operator. Watches how eigendirections shift across decades. Shows how the principal axes of a field change over time, which is distinct from how they respond to semantic abstraction.

## Critique Modes

The third tab group. Expose the tool's own dependencies. Stability, sensitivity, translation, reflexivity.

### Symmetry Breaking Map
Identifies the parameters whose variation splits a unified cloud into distinct clusters. Answers: at what point does this field fragment, and along which axis?

### Phase Diagram
Renders the coarse-graining flow as a 2D projection with basins shaded, fixed points marked, and critical boundaries drawn. Analogous to RG-flow phase diagrams in physics, legible as a single image.

### Embedding Dependence Probe ✅
Re-embeds the same corpus under a second open-weight model chosen from a shortlist (MiniLM, MPNet, BGE) and aligns the two bases via document projections. Dimensions may differ between models; the alignment is computed as the absolute Pearson correlation of per-document scores under each basis, which is basis-dim-independent. Reports per-component agreement (with signed cosine so sign flips are visible), side-by-side loadings, and an overall stability score. The methodological keystone of the Critique tab: every Theoryscope finding is a finding about the corpus-as-measured-by-a-particular-embedding-model, and this probe is how you check whether it survives.

### Perturbation Test ✅
Appends one user-supplied out-of-field text to the corpus, re-embeds that one row, recomputes the eigenbasis on the full set, and aligns to baseline directly in the embedding space. Ships with three curated default probes (commercial press release, medieval canon law, nature writing) so the operation can be run immediately. Returns per-component rotation (1 − |cos|) ranked descending so the user sees which axis was most sensitive to the perturbation, plus where the probe itself lands on each component. A component that rotates substantially under one new paper was not a structural feature of the field.

### Forgetting Curve ✅
Bootstraps over the corpus: at each iteration (default 20), a random fraction of documents (default 20%) is removed, the eigenbasis is recomputed on the survivors, and aligned to baseline. Reports per-component mean / standard deviation / min / interquartile range of the alignment scores across iterations, plus overall stability. The per-iteration table is visible in the deep dive. A formal stability gate for every other finding the tool produces.

### Translated Corpus Probe
Re-runs the eigendecomposition on a machine-translated version of the corpus. Eigendirections that survive translation arguably track concepts. Directions that do not arguably track language-specific framings. The translation is not neutral, and the delta is what matters.

### Corpus-vs-Model Probe
The reflexive operation. Compares the eigendirections of a theoretical corpus against the eigendirections of an open-weight model trained on (or plausibly exposed to) that corpus. Alignment is computed via Procrustes analysis or Canonical Correlation Analysis. The output distinguishes agreement (axes the model preserved), disagreement (axes the corpus has and the model does not), and delta directions (axes the model added from its broader training). 

## Annotations and Atlas

### Critical Annotations Layer
Markdown notes attachable to any eigendirection, fixed point, universality class, or basin. Notes live with the corpus cache and travel with every export. The affordance that distinguishes Theoryscope from a bibliometric dashboard: the critical reading travels with the geometry.

### The Atlas
A curated set of pre-run analyses on named corpora. Each Atlas entry bundles a corpus definition (Zotero query or explicit list), the embedding model, the coarse-graining operator, the results, the stability scores, and David's critical annotations. The Atlas ships with the GitHub release. Three v1 entries are planned: Philosophy of Technology (the Phase 0 corpus), Bias as Universality Class (the AI ethics literature 2015–2025), and a Reflexive Run on the methodology literature that spawned the tool.

## Design Rationale

**Why open-weight embeddings only?** Commercial embedding APIs do not give you what you think they give you. The output-layer vectors of a separately-trained embedding model are not the internal representations of a generative model, and the hidden decisions behind a commercial API (tokeniser, architecture, training data, update schedule) are not available for scholarly inspection. An instrument whose results cannot be reproduced or re-run is not a critical instrument. Theoryscope uses only open-weight embedding models for the same reason Vectorscope does.

**Why implement coarse-graining in process?** A black-box clustering library would produce the same shapes as an in-process implementation, but the black box hides the operator. The critical value of a coarse-graining result depends on the user seeing what has been done to the corpus. Every Theoryscope operator has its source visible, its parameters on screen, and its intermediate states inspectable. Same principle as Vectorscope's Signal Degradation Laboratory, applied to corpora rather than to weights.

**Why RG rather than topic models or citation analysis?** Topic models identify what clusters co-occur in the surface vocabulary. Citation analysis traces reference networks. Both are useful, but both take the surface as their object. Renormalisation group thinking asks a different question: what survives when the surface is abstracted away? The relevant operators, the fixed points, the universality classes are features of the field's geometry under coarse-graining, not of its surface vocabulary. This is the move critics perform intuitively when they say that two traditions converge at depth, and it is the move the instrument makes available for inspection.

**Why provenance with every result?** A finding that cannot be cited is not a finding. The provenance record (corpus hash, document IDs, embedding model, chunking spec, operator, timestamp) is what makes a Theoryscope result something that can travel into a paper. Without it, the geometry is suggestive but not defensible. With it, the reading can be re-run and contested.

**Why tabbed groups rather than a dashboard?** Each operation is a distinct analytical act with its own controls, its own visualisation, and its own deep dive. A dashboard blurs the operations together. A tabbed workspace keeps them honest: one operation, one result, one provenance record. The user chooses which question to ask, and the interface reflects that choice.

**Why stability scores?** Eigendirections depend on the embedding model. Change the model and the computed axes of the field change with it. Reporting a single eigenbasis without its stability under re-embedding is reporting half a result. The Embedding Dependence Probe is not a bonus feature; it is a precondition for taking any of the other operations' outputs seriously.

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) 18+
- Python 3.11+ (for the FastAPI backend and sentence-transformers)
- npm

### Installation

```bash
git clone https://github.com/dmberry/theoryscope.git
cd theoryscope
npm install
```

Set up the Python backend:

```bash
cd backend
./setup.sh
```

This creates a local `.venv` and installs FastAPI, sentence-transformers, scikit-learn, umap-learn, pyzotero, and their dependencies.

### Running

Start the backend:

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

Start the frontend in a separate terminal:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Click **Load Phase 0 Corpus** to embed the 20-document philosophy-of-technology corpus and render the Corpus Map. The deep-dive panel at the bottom of the Corpus Map shows the full provenance record.

## Architecture

```
backend/
  main.py                         # FastAPI app, CORS, route registration
  requirements.txt
  setup.sh                        # venv bootstrap
  corpus/
    loader.py                     # Phase 0 hard-coded corpus; Zotero ingest in later phases
    embed.py                      # Sentence-transformers embedding pipeline
    provenance.py                 # ProvenanceRecord dataclass + corpus hashing
  operations/
    corpus_map.py                 # PCA to 3D with provenance
    (eigendirections, rg_flow, fixed_points, ... stubbed for later phases)
  geometry/                       # stats, reduction helpers (adapted from Vectorscope)
  atlas/
    entries/                      # Pre-computed Atlas entries (later phases)

src/
  app/
    api/backend/[...path]/route.ts  # Next.js proxy to FastAPI on localhost:8000
    layout.tsx                      # Editorial typography and fonts
    page.tsx                        # Tabbed shell: Inspect / Flow / Critique / Annotations / Atlas
    providers.tsx                   # CorpusProvider
  components/
    layout/                         # Header, CorpusLoader, StatusBar
    operations/                     # CorpusMap live; others stubbed
    atlas/                          # AtlasBrowser, AtlasEntry (later phases)
    annotations/                    # AnnotationPanel, AnnotationMarker (later phases)
    viz/                            # CloudScene, EigenBars, FlowAnimation (later phases)
    shared/                         # CorpusSelector, OperatorPicker, ProvenanceBadge
  context/
    CorpusContext.tsx               # Loaded corpus state and load action
  hooks/
    useBackend.ts                   # Thin fetch wrapper for the proxy route
  lib/
    geometry/                       # Copied from Manifold Atlas: cosine, pca, umap, clusters
    provenance.ts                   # Frontend mirror of the backend ProvenanceRecord
    streaming.ts                    # NDJSON streaming (from LLMbench)
    utils.ts
    version.ts
  types/
    corpus.ts
    react-plotly.d.ts               # Type shim for react-plotly.js
```

The architecture mirrors Vectorscope's pattern: a Next.js App Router frontend proxies all `/api/backend/*` requests to a FastAPI backend on `localhost:8000`. No CORS. The Python side owns the embedding model, the corpus cache, and the provenance record; the TypeScript side owns the visualisation and the annotations.

Heavy visualisations (Plotly 3D scatter, and in later phases the Three.js flow animation and phase-diagram renderer) are code-split via `next/dynamic` to keep the initial bundle lean.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 16 (App Router), React 19 |
| Language | TypeScript 5 (strict) |
| Styling | Tailwind CSS 3, editorial design system shared with Vectorscope |
| Visualisation | Plotly.js (GL3D), Three.js (@react-three/fiber), custom SVG |
| Backend | FastAPI, uvicorn, Python 3.11+ |
| Embeddings | sentence-transformers (open-weight, local) |
| Dimensionality Reduction | scikit-learn (PCA), umap-learn |
| Corpus ingestion | pyzotero, networkx (citation graphs) |
| Caching | Local disk (embedded corpora JSON + provenance) |

## Roadmap

### Shipped (v0.5.0)

- [x] Phase 0 scaffold: Next.js frontend, FastAPI backend, editorial design system, hard-coded Philosophy of Technology corpus
- [x] Phase 1: Eigendirections operation with annotated axis cards
- [x] Phase 1.5: Zotero ingestion via pyzotero (user/group libraries, collection picker, credentials in localStorage)
- [x] Phase 1.5: Export pipeline — provenance-stamped JSON downloads from every live operation
- [x] Phase 1.5: On-disk cache for embedded corpora (keyed by document IDs + chunking + embedding model)
- [x] Phase 2: Flow core + three operations (Coarse-Graining Trajectory, Fixed Point Finder, Universality Class Finder) via aggregative k-means coarse-graining
- [x] Phase 3A: Critique tab open — Embedding Dependence Probe, Perturbation Test, Forgetting Curve, all built on a shared eigenbasis-alignment primitive
- [x] Phase 3B: Vector Lab branding — gold accent for inner tier, tool mark, favicon, Help modal
- [x] Phase 4A: Inspect completions — Concept Locator, Author Constellation, Debated-vs-Computed
- [x] Phase 4B: Flow completions — Relevant / Irrelevant Operator Spectrum (variance-preservation scoring across the k-means flow) and Temporal RG Flow (progressively wider year-window aggregation)
- [x] Phase 4C: Critique backends complete — Symmetry Breaking Map, Phase Diagram, Translated Corpus Probe. Routes live on the backend; frontends pending next session. Translated Corpus Probe uses local HelsinkiNLP Marian models (open-weight) with a disk-backed translation cache.

### Next

- [ ] Flow cont.: semantic coarse-graining (local LLM paraphrase) and lexical coarse-graining (WordNet or domain synonyms)
- [ ] Critique frontends: `SymmetryBreaking.tsx`, `PhaseDiagram.tsx`, `TranslatedCorpusProbe.tsx` wired into the existing Critique tab slots
- [ ] Corpus-vs-Model Probe with worked example (AICT corpus vs small open-weight model)
- [ ] Atlas feature with first three entries; Critical Annotations Layer
- [ ] Release blog post on Stunlaw (companion to *Renormalising Theory*)

## Related Work

- Berry, D. M. (2026) 'Renormalising Theory', *Stunlaw*.
- Berry, D. M. (2026) 'What Is Theory Space?', *Stunlaw*.
- Berry, D. M. (2026) 'Vector Theory', *Stunlaw*. Available at: https://stunlaw.blogspot.com/2026/02/vector-theory.html
- Berry, D. M. (2026) *Artificial Intelligence and Critical Theory*. MUP.
- Fisher, M. E. (1974) 'The renormalization group in the theory of critical behavior', *Reviews of Modern Physics*, 46(4), pp. 597-616.
- Mehta, P. and Schwab, D. J. (2014) 'An exact mapping between the variational renormalization group and deep learning'. Available at: https://arxiv.org/abs/1410.3831.

## Acknowledgements

Concept and Design by David M. Berry, implemented with Claude Code 4.6. Design system adapted from the [CCS Workbench](https://github.com/dmberry/ccs-wb). Theoryscope shares its editorial design system, geometry library, NDJSON streaming, and FastAPI proxy pattern with its sibling instruments [Vectorscope](https://github.com/dmberry/vectorscope), [Manifold Atlas](https://github.com/dmberry/manifold-atlas), and [LLMbench](https://github.com/dmberry/LLMbench).

Many thanks to John Hessler for the renormalisation group and deep-learning genealogy reading list that inspired the Theoryscope.

## Licence

MIT
