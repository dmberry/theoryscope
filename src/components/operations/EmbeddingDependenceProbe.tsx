"use client";

import { useCallback, useEffect, useState } from "react";
import { backendGet, backendPost } from "@/hooks/useBackend";
import { ExportButton } from "@/components/shared/ExportButton";
import { StabilityBar } from "@/components/viz/StabilityBar";
import { useCorpusSource } from "@/context/CorpusSourceContext";
import type {
  EmbeddingModelInfo,
  EmbeddingProbeResponse,
  ProbeBasis,
} from "@/types/critique";

type RunState = "idle" | "loading" | "ready" | "error";

export function EmbeddingDependenceProbe() {
  const { buildPayload, selected, zoteroReady } = useCorpusSource();
  const [models, setModels] = useState<EmbeddingModelInfo[]>([]);
  const [probeModel, setProbeModel] = useState<string>("");
  const [nComponents, setNComponents] = useState<number>(5);
  const [state, setState] = useState<RunState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<EmbeddingProbeResponse | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const result = await backendGet<{ models: EmbeddingModelInfo[] }>(
          "/embedding-probe/models",
        );
        setModels(result.models);
        // Default to the first non-baseline (second in the list).
        if (result.models.length > 1) setProbeModel(result.models[1].model_id);
      } catch {
        // Silent: the backend may not be running yet; the Run button will surface the error.
      }
    })();
  }, []);

  const canRun =
    state !== "loading" &&
    probeModel.length > 0 &&
    (selected.kind === "hardcoded" || zoteroReady);

  const run = useCallback(async () => {
    setState("loading");
    setError(null);
    try {
      const result = await backendPost<EmbeddingProbeResponse>(
        "/embedding-probe",
        {
          corpus: buildPayload(),
          probe_model_id: probeModel,
          n_components: nComponents,
          n_loadings: 3,
        },
      );
      setData(result);
      setState("ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setState("error");
    }
  }, [buildPayload, probeModel, nComponents]);

  return (
    <section className="space-y-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h2 className="font-display text-xl text-ink">
            Embedding Dependence Probe
          </h2>
          <p className="text-sm text-ink/70 mt-1 max-w-3xl">
            Re-run the eigendecomposition under a second open-weight
            embedding model and report the agreement with the baseline.
            Eigendirections that survive a change of embedding are
            stronger claims about the field than eigendirections that
            do not. This is the methodological keystone of the Critique
            tab.
          </p>
        </div>
        <ExportButton payload={data} filename="theoryscope-embedding-probe" />
      </header>

      <div className="flex flex-wrap items-end gap-6 p-4 border border-ink/10 bg-ivory/40">
        <label className="flex flex-col text-xs text-ink/70 gap-1">
          <span className="uppercase tracking-widest text-ink/50">
            Probe model
          </span>
          <select
            value={probeModel}
            onChange={(e) => setProbeModel(e.target.value)}
            className="px-2 py-1.5 bg-white/60 border border-ink/20 text-sm"
          >
            {models.length === 0 ? (
              <option value="">(load backend first)</option>
            ) : null}
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {m.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col text-xs text-ink/70 gap-1">
          <span className="uppercase tracking-widest text-ink/50">
            Components
          </span>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={2}
              max={10}
              step={1}
              value={nComponents}
              onChange={(e) => setNComponents(Number(e.target.value))}
              className="w-40 accent-burgundy"
            />
            <span className="font-mono text-ink tabular-nums w-6 text-right">
              {nComponents}
            </span>
          </div>
        </label>
        <button
          type="button"
          onClick={() => void run()}
          disabled={!canRun}
          className="ml-auto px-3 py-1.5 bg-burgundy text-ivory text-sm tracking-wide uppercase disabled:opacity-50 hover:bg-burgundy-700 transition-colors"
        >
          {state === "loading" ? "Probing…" : "Run probe"}
        </button>
      </div>

      {state === "loading" ? (
        <div className="p-8 border border-ink/10 bg-ivory/50 text-ink/70">
          Embedding corpus under the probe model (first run may download
          the model — this can take a minute)…
        </div>
      ) : null}

      {state === "error" ? (
        <div className="p-6 border border-rose-300 bg-rose-50/40 text-rose-800 text-sm">
          {error ?? "Unknown error."}
        </div>
      ) : null}

      {data ? <ProbeResults data={data} /> : null}
    </section>
  );
}

function ProbeResults({ data }: { data: EmbeddingProbeResponse }) {
  return (
    <div className="space-y-6">
      <div className="border border-ink/10 bg-white/60 p-4 space-y-3">
        <div className="flex items-baseline justify-between">
          <h3 className="font-display text-base text-ink">Overall stability</h3>
          <span className="text-xs font-mono text-ink/50">
            mean |cos| across matched components
          </span>
        </div>
        <StabilityBar value={data.alignment.stability} label="stability" />
        <div className="text-xs text-ink/70 max-w-prose">
          <strong>Baseline:</strong> {data.baseline.model_id} · dim{" "}
          {data.baseline.dimension}
          <br />
          <strong>Probe:</strong> {data.probe.model_id} · dim{" "}
          {data.probe.dimension}
        </div>
      </div>

      <div>
        <h3 className="font-display text-base text-ink mb-3">
          Per-component agreement
        </h3>
        <div className="space-y-3">
          {data.alignment.matches.map((m) => (
            <div
              key={`pc-${m.a_index}`}
              className="border border-ink/10 bg-white/50 p-3 space-y-2"
            >
              <div className="flex items-baseline justify-between">
                <span className="font-display text-sm text-ink">
                  Baseline PC{m.a_index + 1} ↔ Probe PC{m.b_index + 1}
                </span>
                <span className="text-xs font-mono text-ink/60">
                  signed cos = {m.signed_cosine.toFixed(3)}
                </span>
              </div>
              <StabilityBar value={m.abs_cosine} />
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <BasisPanel basis={data.baseline} title="Baseline basis" />
        <BasisPanel basis={data.probe} title="Probe basis" />
      </div>

      <details className="border border-ink/10 bg-ivory/40 p-4">
        <summary className="cursor-pointer text-sm font-semibold text-ink">
          Deep dive · provenance
        </summary>
        <pre className="mt-3 overflow-x-auto text-xs font-mono text-ink/80">
{JSON.stringify(data.provenance, null, 2)}
        </pre>
      </details>
    </div>
  );
}

function BasisPanel({ basis, title }: { basis: ProbeBasis; title: string }) {
  return (
    <div className="border border-ink/10 bg-white/60 p-4 space-y-3">
      <div className="flex items-baseline justify-between">
        <h4 className="font-display text-sm text-ink">{title}</h4>
        <span className="text-xs font-mono text-ink/50">
          {basis.model_id.split("/").pop()} · dim {basis.dimension}
        </span>
      </div>
      <ol className="space-y-3">
        {basis.loadings.map((componentLoadings, pcIdx) => {
          const variance = basis.variance_explained[pcIdx] ?? 0;
          const positives = componentLoadings.filter((l) => l.pole === "positive");
          const negatives = componentLoadings.filter((l) => l.pole === "negative");
          return (
            <li key={`pc-${pcIdx}`} className="text-xs space-y-1">
              <div className="flex items-baseline justify-between">
                <span className="font-display text-ink">PC{pcIdx + 1}</span>
                <span className="font-mono text-ink/60">
                  {(variance * 100).toFixed(1)}%
                </span>
              </div>
              <div className="text-emerald-800">
                +{" "}
                {positives
                  .map((l) => `${l.author.split(",")[0]} ${l.year}`)
                  .join(" · ")}
              </div>
              <div className="text-rose-800">
                −{" "}
                {negatives
                  .map((l) => `${l.author.split(",")[0]} ${l.year}`)
                  .join(" · ")}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
