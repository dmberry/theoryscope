import type { ProvenanceRecord } from "../lib/provenance";
import type { CorpusDocument } from "./corpus";

/** One coarse-graining step, suitable for frontend animation. */
export type FlowStep = {
  step: number;
  k: number;
  labels: number[];                 // n_docs, each entry ∈ [0, k)
  doc_coords_2d: [number, number][]; // n_docs × 2
};

export type CoarseGrainingTrajectoryResponse = {
  documents: CorpusDocument[];
  schedule: number[];
  pca2d_variance: [number, number];
  steps: FlowStep[];
  provenance: ProvenanceRecord;
};

export type BasinEntry = {
  basin_index: number;
  size: number;
  exemplar: CorpusDocument | null;
  members: CorpusDocument[];
  centroid_2d: [number, number];
};

export type FixedPointsResponse = {
  documents: CorpusDocument[];
  schedule: number[];
  n_basins: number;
  basins: BasinEntry[];
  terminal_coords_2d: [number, number][];
  terminal_labels: number[];
  provenance: ProvenanceRecord;
};

export type UniversalityClass = {
  class_index: number;
  size: number;
  members: CorpusDocument[];
  /**
   * Mean cosine similarity between members at their surface (pre-flow)
   * positions. Lower values indicate universality: surface-diverse
   * positions converging on the same terminal basin.
   */
  surface_mean_cosine: number;
};

export type UniversalityClassesResponse = {
  documents: CorpusDocument[];
  schedule: number[];
  n_classes: number;
  classes: UniversalityClass[];
  initial_coords_2d: [number, number][];
  terminal_labels: number[];
  provenance: ProvenanceRecord;
};

export type FlowRequest = {
  corpus: unknown;
  n_steps: number;
  seed: number;
};
