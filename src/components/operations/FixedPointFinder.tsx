"use client";

import { useCallback, useMemo, useState } from "react";
import { backendPost } from "@/hooks/useBackend";
import { ExportButton } from "@/components/shared/ExportButton";
import { FlowScatter } from "@/components/viz/FlowScatter";
import { useCorpusSource } from "@/context/CorpusSourceContext";
import type { FixedPointsResponse } from "@/types/flow";

type RunState = "idle" | "loading" | "ready" | "error";

export function FixedPointFinder() {
  const { buildPayload, selected, zoteroReady } = useCorpusSource();
  const [state, setState] = useState<RunState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<FixedPointsResponse | null>(null);
  const [nSteps, setNSteps] = useState<number>(6);
  const [focusBasin, setFocusBasin] = useState<number | null>(null);

  const canRun =
    state !== "loading" && (selected.kind === "hardcoded" || zoteroReady);

  const run = useCallback(async () => {
    setState("loading");
    setError(null);
    try {
      const result = await backendPost<FixedPointsResponse>("/fixed-points", {
        corpus: buildPayload(),
        n_steps: nSteps,
        seed: 0,
      });
      setData(result);
      setFocusBasin(null);
      setState("ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setState("error");
    }
  }, [buildPayload, nSteps]);

  const highlightIndices = useMemo(() => {
    if (!data || focusBasin === null) return undefined;
    const hits: number[] = [];
    data.terminal_labels.forEach((lbl, i) => {
      if (lbl === focusBasin) hits.push(i);
    });
    return hits;
  }, [data, focusBasin]);

  const paletteSize = useMemo(() => (data ? data.n_basins : 1), [data]);

  return (
    <section className="space-y-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h2 className="font-display text-xl text-ink">Fixed Point Finder</h2>
          <p className="text-sm text-ink/70 mt-1 max-w-3xl">
            Iterate the aggregative k-means flow to its terminal step and
            report the basins every document falls into. Basins are the
            fixed points of this flow: positions that do not move under
            further coarse-graining (within the terminal cluster count).
            Click a basin row to highlight its members on the terminal
            map.
          </p>
        </div>
        <ExportButton payload={data} filename="theoryscope-fixed-points" />
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
          {state === "loading" ? "Finding…" : "Find fixed points"}
        </button>
      </div>

      {state === "loading" ? (
        <div className="p-8 border border-ink/10 bg-ivory/50 text-ink/70">
          Running the flow to convergence…
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
              Terminal map · {data.n_basins} basin
              {data.n_basins === 1 ? "" : "s"}
            </h3>
            {focusBasin !== null ? (
              <button
                type="button"
                onClick={() => setFocusBasin(null)}
                className="text-xs underline decoration-ink/30 hover:decoration-ink text-ink/70"
              >
                clear highlight
              </button>
            ) : null}
          </div>

          <FlowScatter
            coords={data.terminal_coords_2d}
            labels={data.terminal_labels}
            documents={data.documents}
            paletteSize={paletteSize}
            highlightIndices={highlightIndices}
            height={420}
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {data.basins.map((b) => (
              <button
                key={b.basin_index}
                type="button"
                onClick={() =>
                  setFocusBasin(
                    focusBasin === b.basin_index ? null : b.basin_index,
                  )
                }
                className={[
                  "text-left p-4 border transition-colors",
                  focusBasin === b.basin_index
                    ? "border-burgundy bg-white/70"
                    : "border-ink/10 bg-white/50 hover:bg-ivory/50",
                ].join(" ")}
              >
                <div className="flex items-baseline justify-between">
                  <span className="font-display text-sm text-ink">
                    Basin {b.basin_index + 1}
                  </span>
                  <span className="text-xs font-mono text-ink/60">
                    {b.size} document{b.size === 1 ? "" : "s"}
                  </span>
                </div>
                {b.exemplar ? (
                  <div className="mt-1 text-xs text-ink/70">
                    Exemplar: {b.exemplar.author} {b.exemplar.year}
                    <span className="ml-1 text-ink/50">— {b.exemplar.title}</span>
                  </div>
                ) : null}
                <ul className="mt-2 text-xs text-ink/60 space-y-0.5">
                  {b.members.map((m) => (
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
