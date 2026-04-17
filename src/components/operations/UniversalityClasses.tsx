"use client";

import { useCallback, useMemo, useState } from "react";
import { backendPost } from "@/hooks/useBackend";
import { ExportButton } from "@/components/shared/ExportButton";
import { FlowScatter } from "@/components/viz/FlowScatter";
import { useCorpusSource } from "@/context/CorpusSourceContext";
import type { UniversalityClassesResponse } from "@/types/flow";

type RunState = "idle" | "loading" | "ready" | "error";

export function UniversalityClasses() {
  const { buildPayload, selected, zoteroReady } = useCorpusSource();
  const [state, setState] = useState<RunState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<UniversalityClassesResponse | null>(null);
  const [nSteps, setNSteps] = useState<number>(6);
  const [focusClass, setFocusClass] = useState<number | null>(null);

  const canRun =
    state !== "loading" && (selected.kind === "hardcoded" || zoteroReady);

  const run = useCallback(async () => {
    setState("loading");
    setError(null);
    try {
      const result = await backendPost<UniversalityClassesResponse>(
        "/universality-classes",
        {
          corpus: buildPayload(),
          n_steps: nSteps,
          seed: 0,
        },
      );
      setData(result);
      setFocusClass(null);
      setState("ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setState("error");
    }
  }, [buildPayload, nSteps]);

  const highlightIndices = useMemo(() => {
    if (!data || focusClass === null) return undefined;
    const hits: number[] = [];
    data.terminal_labels.forEach((lbl, i) => {
      if (lbl === focusClass) hits.push(i);
    });
    return hits;
  }, [data, focusClass]);

  // Rank classes by descending universality: a class whose surface-mean
  // cosine similarity is low gathers surface-different positions under
  // the flow, which is the universality-class finding worth inspecting.
  const ranked = useMemo(() => {
    if (!data) return [];
    return [...data.classes].sort(
      (a, b) => a.surface_mean_cosine - b.surface_mean_cosine,
    );
  }, [data]);

  return (
    <section className="space-y-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h2 className="font-display text-xl text-ink">Universality Classes</h2>
          <p className="text-sm text-ink/70 mt-1 max-w-3xl">
            Cluster documents by the basin they flow to, rather than by
            their surface embedding. A class whose members were surface-
            different but converge on the same basin is a universality-
            class finding worth reading. The ranking below puts the
            lowest surface-similarity classes first.
          </p>
        </div>
        <ExportButton
          payload={data}
          filename="theoryscope-universality-classes"
        />
      </header>

      <div className="flex flex-wrap items-end gap-6 p-4 border border-ink/10 bg-ivory/40">
        <label className="flex flex-col text-xs text-ink/70 gap-1">
          <span className="uppercase tracking-widest text-ink/50">Steps</span>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={3}
              max={10}
              step={1}
              value={nSteps}
              onChange={(e) => setNSteps(Number(e.target.value))}
              className="w-40 accent-burgundy"
            />
            <span className="font-mono text-ink tabular-nums w-6 text-right">
              {nSteps}
            </span>
          </div>
        </label>
        <button
          type="button"
          onClick={() => void run()}
          disabled={!canRun}
          className="ml-auto px-3 py-1.5 bg-burgundy text-ivory text-sm tracking-wide uppercase disabled:opacity-50 hover:bg-burgundy-700 transition-colors"
        >
          {state === "loading" ? "Clustering…" : "Find universality classes"}
        </button>
      </div>

      {state === "loading" ? (
        <div className="p-8 border border-ink/10 bg-ivory/50 text-ink/70">
          Running flow and clustering by terminal basin…
        </div>
      ) : null}

      {state === "error" ? (
        <div className="p-6 border border-rose-300 bg-rose-50/40 text-rose-800 text-sm">
          {error ?? "Unknown error."}
        </div>
      ) : null}

      {data ? (
        <div className="space-y-6">
          <div className="flex items-baseline justify-between">
            <h3 className="font-display text-base text-ink">
              Surface map · coloured by terminal class
            </h3>
            {focusClass !== null ? (
              <button
                type="button"
                onClick={() => setFocusClass(null)}
                className="text-xs underline decoration-ink/30 hover:decoration-ink text-ink/70"
              >
                clear highlight
              </button>
            ) : null}
          </div>

          <FlowScatter
            coords={data.initial_coords_2d}
            labels={data.terminal_labels}
            documents={data.documents}
            paletteSize={data.n_classes}
            highlightIndices={highlightIndices}
            height={420}
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {ranked.map((c) => (
              <button
                key={c.class_index}
                type="button"
                onClick={() =>
                  setFocusClass(
                    focusClass === c.class_index ? null : c.class_index,
                  )
                }
                className={[
                  "text-left p-4 border transition-colors",
                  focusClass === c.class_index
                    ? "border-burgundy bg-white/70"
                    : "border-ink/10 bg-white/50 hover:bg-ivory/50",
                ].join(" ")}
              >
                <div className="flex items-baseline justify-between">
                  <span className="font-display text-sm text-ink">
                    Class {c.class_index + 1}
                  </span>
                  <span className="text-xs font-mono text-ink/60 tabular-nums">
                    {c.size} · surface cos ={" "}
                    {c.surface_mean_cosine.toFixed(3)}
                  </span>
                </div>
                <UniversalityBar cos={c.surface_mean_cosine} />
                <ul className="mt-2 text-xs text-ink/60 space-y-0.5">
                  {c.members.map((m) => (
                    <li key={m.id} className="truncate">
                      {m.author} {m.year} — {m.title}
                    </li>
                  ))}
                </ul>
              </button>
            ))}
          </div>

          <details className="border border-ink/10 bg-ivory/40 p-4">
            <summary className="cursor-pointer text-sm font-semibold text-ink">
              Deep dive · schedule + provenance
            </summary>
            <div className="mt-4 space-y-3 text-xs text-ink/80">
              <p className="font-mono">
                Schedule: {data.schedule.join(" → ")}
              </p>
              <p className="text-ink/70 max-w-prose">
                Surface mean cosine is the average pairwise cosine
                similarity between members at their pre-flow embedding
                positions. Low values indicate a universality class:
                positions that look different at the surface but flow
                to the same basin. High values indicate a class that
                the flow has preserved rather than created.
              </p>
              <pre className="overflow-x-auto font-mono bg-ivory/40 p-3 border border-ink/10">
{JSON.stringify(data.provenance, null, 2)}
              </pre>
            </div>
          </details>
        </div>
      ) : null}
    </section>
  );
}

function UniversalityBar({ cos }: { cos: number }) {
  // Map cos similarity [0, 1] to a universality score [0, 1] where
  // universality = 1 - cos (lower cos ⇒ higher universality).
  const universality = Math.max(0, Math.min(1, 1 - cos));
  const pct = universality * 100;
  return (
    <div
      className="mt-2 h-1 bg-ink/10 overflow-hidden"
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pct)}
      title={`Universality ≈ ${pct.toFixed(0)}%`}
    >
      <div
        className="h-full bg-burgundy"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}
