import type { ProvenanceRecord } from "../lib/provenance";
import type { CorpusDocument } from "./corpus";

/* ------------------------ Embedding Dependence Probe --------------------- */

export type EmbeddingModelInfo = {
  model_id: string;
  label: string;
  dimension: number;
};

export type ProbeLoading = {
  id: string;
  author: string;
  year: number;
  title: string;
  score: number;
  pole: "positive" | "negative";
};

export type ProbeBasis = {
  model_id: string;
  dimension: number;
  variance_explained: number[];
  loadings: ProbeLoading[][]; // per-component, ranked [+pos..., -neg...]
};

export type ComponentMatch = {
  a_index: number;
  b_index: number;
  abs_cosine: number;
  signed_cosine: number;
};

export type AlignmentPayload = {
  matches: ComponentMatch[];
  per_component: number[];
  stability: number;
  per_component_rotation?: number[];
  ranked_by_rotation?: number[];
};

export type EmbeddingProbeResponse = {
  documents: CorpusDocument[];
  baseline: ProbeBasis;
  probe: ProbeBasis;
  alignment: AlignmentPayload;
  provenance: ProvenanceRecord;
};

/* ---------------------------- Perturbation Test -------------------------- */

export type PerturbationResponse = {
  documents: CorpusDocument[];
  baseline: { variance_explained: number[] };
  perturbed: { variance_explained: number[] };
  alignment: AlignmentPayload;
  probe: {
    label: string;
    char_length: number;
    projection_on_perturbed_basis: number[];
  };
  provenance: ProvenanceRecord;
};

/* --------------------------- Forgetting Curve ---------------------------- */

export type ForgettingCurveResponse = {
  documents: CorpusDocument[];
  n_components: number;
  n_iterations: number;
  drop_fraction: number;
  per_pc_mean: number[];
  per_pc_std: number[];
  per_pc_p25: number[];
  per_pc_p75: number[];
  per_pc_min: number[];
  per_iteration: number[][]; // (n_iterations, n_components)
  per_iteration_stability: number[];
  overall_stability: number;
  provenance: ProvenanceRecord;
};
