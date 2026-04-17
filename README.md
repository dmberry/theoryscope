# Theoryscope

**A renormalisation-group workbench for corpora of theoretical texts.**

**Author:** David M. Berry
**Institution:** University of Sussex
**Version:** 0.2.0
**Date:** 17 April 2026
**Licence:** MIT

Theoryscope is a web-based research tool that treats a corpus of theoretical texts as a geometry and asks renormalisation-group and eigenvector questions about it. It ingests a corpus, embeds each text with an open-weight model, and provides operations for inspecting the cloud's structure using vocabulary drawn from renormalisation group theory and linear algebra: principal eigendirections of a field, coarse-graining flows, fixed points, relevant and irrelevant operators, and universality classes.

The tool is designed for critical inquiry into the geometry of intellectual fields, not bibliometric measurement. Where existing computational literature tools (citation graph analyses, topic models, bibliometric dashboards) count and cluster surface features, Theoryscope provides an instrument for making visible the gap between what a field says it is about and what its geometry suggests it is about.

> Theoryscope is part of the [Vector Lab](https://github.com/dmberry) family of research instruments, alongside [Manifold Atlas](https://github.com/dmberry/manifold-atlas), [Vectorscope](https://github.com/dmberry/vectorscope), and [LLMbench](https://github.com/dmberry/LLMbench). The four tools share an editorial design system, an open-weight-only methodology, and a commitment to making the geometry of meaning legible for critical analysis. They diverge in their object: Manifold Atlas compares output geometries between models, Vectorscope inspects the internals of a single open-weight model, LLMbench reads the surface of model outputs as prose, and Theoryscope maps the geometry of a corpus of theoretical texts.

## Scholarly Context

Theoryscope emerges from the convergence of three research programmes.

**The Vector Lab.** The tool is the fourth instrument in a family alongside Manifold Atlas (between-model comparison of output embeddings), Vectorscope (within-model inspection of open-weight internals), and LLMbench (close reading of model outputs). Manifold Atlas and Vectorscope take a model as their object; Theoryscope takes a *corpus* as its object, extending the Vector Lab's critical posture beyond the machine to the thought that is written about the machine. All four instruments share the editorial design system, an open-weight-only methodology (following Castelle), and a commitment to making every parameter visible on screen rather than hidden behind opaque kernels.

**The renormalisation group as critical method.** Berry (2026) *Renormalising Theory* argues that claims critics already perform intuitively, that "Foucault, Deleuze and Guattari really come down to X", or that "Habermas and Rorty converge at depth", are in fact renormalisation-group claims about what survives when a field is coarse-grained. The physics vocabulary of flow on theory space (Fisher 1974), relevant and irrelevant operators, universality classes, and the mapping to deep learning (Mehta and Schwab 2014; Lin, Tegmark and Rolnick 2017) provides a formal language for a critical move that has always happened through the reading. Theoryscope is the instrument that follows from the argument.

**Counterfactual media and the theory-space concept.** Berry (2026) *What Is Theory Space?* treats theory space as the inaccessible totality of counterfactual media from which a particular manifold is selected during training. That post is about what cannot be inspected. Theoryscope works on a cognate but distinct object, a corpus of theoretical texts that is empirically given. The kinship is methodological. The difference is that Theoryscope's corpus space is available for inspection, coarse-graining, and re-embedding. It is the empirical companion to the conceptual argument.

## Corpora and Embeddings: A Primer

Theoryscope uses two kinds of object, and it helps to understand the difference.

A **corpus** is a finite, explicitly defined set of documents: a Zotero collection, an author's complete works, a time-bounded selection of a field. A corpus is a theoretical claim in its own right. Who is in it and who is out, under what criterion, with what filters, determines what the instrument can see. Theoryscope records the corpus definition as part of every result so that findings can be reproduced and contested.

An **embedding model** (MiniLM, BGE, Nomic, Qwen3-Embedding) takes text in and produces a **vector** out: a list of numbers (typically 384 to 1024 floating-point values). No language comes back, just coordinates in a high-dimensional space. Those coordinates are the model's geometric encoding of what the text "means", where meaning is reduced to position. The embedding model is the telescope: it converts text into a location.

This matters for Theoryscope because every operation is sensitive to which embedding model is computing the distances. The same corpus, embedded by two different open-weight models, will not produce the same eigendirections. The Embedding Dependence Probe exists to make this visible. An operation that reports a single eigenbasis without its stability under re-embedding is reporting half a result. Following Michael Castelle's correspondence on the critical limits of commercial embedding APIs, Theoryscope uses only open-weight embedding models, whose architectures, training data, and update schedules are available for scholarly inspection.

The difference between a **chat model** and an **embedding model** is relevant here too. A chat model (GPT, Claude, Llama) takes text in and produces text out. An embedding model stops earlier: it produces the vector and hands it to you. This is why embedding API calls are cheap (fractions of a penny) while chat API calls are expensive. Theoryscope uses embedding models as its core instrument. Chat models appear only in later phases for the semantic coarse-graining operator, where an LLM is used to produce progressively more abstract paraphrases of a text before re-embedding.

## Operations at a Glance

Theoryscope is organised as a tabbed workspace with five groups: three operational groups (Inspect, Flow, Critique) and two cross-cutting features (Annotations, Atlas).

| Group | Operation | Core question |
|---|---|---|
| **Inspect** | Corpus Map | What does the cloud look like? |
| Inspect | Eigendirections | What are the principal axes of the field? |
| Inspect | Concept Locator | Where does this concept sit, and along which axis does it vary? |
| Inspect | Author Constellation | How do authors differ and where do they concentrate? |
| **Flow** | Coarse-Graining Trajectory | How do points move under successive coarse-grainings? |
| Flow | Fixed Point Finder | Where does the flow stop? |
| Flow | Relevant / Irrelevant Operator Spectrum | Which concepts persist and which wash out? |
| Flow | Universality Class Finder | Which starting positions converge to the same endpoint? |
| Flow | Temporal RG Flow | How do the principal axes shift across decades? |
| **Critique** | Symmetry Breaking Map | Where does the field fragment, and along which axis? |
| Critique | Phase Diagram | What does the flow field look like as a whole? |
| Critique | Embedding Dependence Probe | Do these results survive a different embedding? |
| Critique | Perturbation Test | How robust is this result to a small change in the corpus? |
| Critique | Forgetting Curve | How fragile are these eigendirections under bootstrap? |
| Critique | Translated Corpus Probe | What survives translation? |
| Critique | Corpus-vs-Model Probe | Do the eigendirections of a corpus match those of a model trained on it? |
| **Annotations** | Critical Annotations Layer | Markdown notes attached to any geometric feature, exported with the result. |
| **Atlas** | Curated pre-run analyses on named corpora, with provenance and annotations. |

All operations carry a provenance record (corpus hash, embedding model, chunking spec, coarse-graining operator, optional stability score) that travels with every export.

Phase 0 ships with Corpus Map only. The other operations are stubbed and land in subsequent phases.

## Inspect Mode

The opening workspace. Load a corpus, watch it appear as geometry, read its principal axes.

### Core features

- **Open-weight embeddings.** Following the Castelle critique, Theoryscope does not use commercial embedding APIs. Phase 0 uses `sentence-transformers/all-MiniLM-L6-v2` as the default embedder. Later phases will delegate to a running Vectorscope session when one is available, and will support comparative runs under multiple embedding models.
- **Corpus as a cloud.** Each document is a point in a high-dimensional embedding space. The Corpus Map projects to three dimensions via PCA and renders the cloud as a 3D scatter in Plotly, with documents coloured by year and labelled by author.
- **Annotated eigendirections.** Each principal component is annotated with the documents that load most positively and most negatively on it. The critical use is comparison with the field's debated oppositions: computed axes and debated axes rarely match, and the gap is diagnostic.
- **Concept locator.** Enter a short text or concept. The tool embeds it into the same space, returns the nearest documents and authors, and reports the eigenvector most aligned with the query.
- **Author constellations.** Aggregate an author's corpus to a single centroid with intra-author spread. Overlay multiple authors on the same eigenbasis to compare their constellations.
- **Provenance record.** Every Corpus Map view has a deep-dive panel exposing the full provenance JSON: corpus source, document IDs, embedding model and dimension, chunking spec, ingestion timestamp, and SHA-256 corpus hash.

## Flow Modes

The second tab group. Coarse-grain the corpus, watch it flow, locate its fixed points.

### Coarse-Graining Trajectory
Applies a chosen coarse-graining operator in stepwise passes, animating how points move. The operator is visible on screen with its parameters. Four operators are supported: aggregative (documents to authors to traditions), semantic (LLM-mediated paraphrase at increasing abstraction), lexical (technical vocabulary to everyday synonyms), and citation-graph (documents that cite each other as candidates for aggregative merging). Phase 0 implements aggregative only; the others land in later phases.

### Fixed Point Finder
Iterates the coarse-graining until positions stop moving within a tolerance. Returns the fixed points and the basin of attraction each starting point fell into. Basins are rendered as shaded regions of the corpus map, fixed points marked and labelled.

### Relevant / Irrelevant Operator Spectrum
For a user-supplied or auto-extracted list of concepts, ranks each by whether its role in discriminating texts grows or shrinks under coarse-graining. Relevant operators persist at every scale; irrelevant operators wash out. The critical interpretation is the reader's; the tool reports the pattern.

### Universality Class Finder
Clusters corpus points by the fixed point they flow to, rather than by surface embedding. Highlights hidden affinities (positions that look opposed but converge) and hidden divergences (positions that look similar but diverge).

### Temporal RG Flow
Coarse-grains the corpus by time window rather than by semantic or aggregative operator. Watches how eigendirections shift across decades. Shows how the principal axes of a field change over time, which is distinct from how they respond to semantic abstraction.

## Critique Modes

The third tab group. Expose the tool's own dependencies. Stability, sensitivity, translation, reflexivity.

### Symmetry Breaking Map
Identifies the parameters whose variation splits a unified cloud into distinct clusters. Answers: at what point does this field fragment, and along which axis?

### Phase Diagram
Renders the coarse-graining flow as a 2D projection with basins shaded, fixed points marked, and critical boundaries drawn. Analogous to RG-flow phase diagrams in physics, legible as a single image.

### Embedding Dependence Probe
Recomputes the corpus map, the eigendirections, and the RG flow under a different open-weight embedding model. Reports the delta. Stability across embeddings is a stronger claim about the field than any single run.

### Perturbation Test
Adds one out-of-field paper to the corpus. Measures how far the top eigenvectors move. A diagnostic for how robust a particular eigendirection is to a small change in what is in the corpus.

### Forgetting Curve
Removes a random 20% of the corpus, redoes the eigendecomposition, and compares. Iterates. Reports which eigendirections are robust under corpus resampling and which are fragile.

### Translated Corpus Probe
Re-runs the eigendecomposition on a machine-translated version of the corpus. Eigendirections that survive translation arguably track concepts. Directions that do not arguably track language-specific framings. The translation is not neutral, and the delta is what matters.

### Corpus-vs-Model Probe
The reflexive operation. Compares the eigendirections of a theoretical corpus against the eigendirections of an open-weight model trained on (or plausibly exposed to) that corpus. Alignment is computed via Procrustes analysis or Canonical Correlation Analysis. The output distinguishes agreement (axes the model preserved), disagreement (axes the corpus has and the model does not), and delta directions (axes the model added from its broader training). This is the Castelle-level methodological question operationalised, and the operation that most directly opens onto the vector theory book's argument about the relationship between the medium and the theories it has been trained on.

## Annotations and Atlas

### Critical Annotations Layer
Markdown notes attachable to any eigendirection, fixed point, universality class, or basin. Notes live with the corpus cache and travel with every export. The affordance that distinguishes Theoryscope from a bibliometric dashboard: the critical reading travels with the geometry.

### The Atlas
A curated set of pre-run analyses on named corpora. Each Atlas entry bundles a corpus definition (Zotero query or explicit list), the embedding model, the coarse-graining operator, the results, the stability scores, and David's critical annotations. The Atlas ships with the GitHub release. Three v1 entries are planned: Philosophy of Technology (the Phase 0 corpus), Bias as Universality Class (the AI ethics literature 2015–2025), and a Reflexive Run on the methodology literature that spawned the tool.

## Design Rationale

**Why open-weight embeddings only?** Commercial embedding APIs do not give you what you think they give you. As Michael Castelle has argued, the output-layer vectors of a separately-trained embedding model are not the internal representations of a generative model, and the hidden decisions behind a commercial API (tokeniser, architecture, training data, update schedule) are not available for scholarly inspection. An instrument whose results cannot be reproduced or re-run is not a critical instrument. Theoryscope uses only open-weight embedding models for the same reason Vectorscope does.

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

- [x] Phase 0 scaffold: Next.js frontend, FastAPI backend, editorial design system, hard-coded Philosophy of Technology corpus, Corpus Map operation end to end
- [x] Phase 1: Eigendirections operation with annotated axis cards (positive and negative poles per component)
- [x] Phase 1.5: Zotero ingestion via pyzotero (user/group libraries, collection picker, credentials in localStorage)
- [x] Phase 1.5: Export pipeline — provenance-stamped JSON downloads from Corpus Map and Eigendirections
- [x] Phase 1.5: On-disk cache for embedded corpora (keyed by document IDs + chunking + embedding model)
- [ ] Phase 1.5: Debated-vs-computed comparison UI (deferred from Phase 1)
- [x] Phase 2: Flow core + three operations (Coarse-Graining Trajectory, Fixed Point Finder, Universality Class Finder) via aggregative k-means coarse-graining
- [ ] Phase 2 cont.: Relevant / Irrelevant Operator Spectrum, Temporal RG Flow, semantic and lexical coarse-graining operators
- [ ] Phase 3: Critique operations (Symmetry Breaking, Phase Diagram rendering, Embedding Dependence Probe, Perturbation Test, Forgetting Curve, Translated Corpus Probe)
- [ ] Phase 4: Corpus-vs-Model Probe with worked example (AICT corpus vs small open-weight model); Atlas feature with first three entries; Critical Annotations Layer
- [ ] Phase 5: Release blog post on Stunlaw (companion to *Renormalising Theory*)

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
