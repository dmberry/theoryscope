"use client";

import { useCallback, useMemo, useState } from "react";
import { backendPost } from "@/hooks/useBackend";
import { ExportButton } from "@/components/shared/ExportButton";
import { StabilityBar } from "@/components/viz/StabilityBar";
import { useCorpusSource } from "@/context/CorpusSourceContext";
import type { ForgettingCurveResponse } from "@/types/critique";

type RunState = "idle" | "loading" | "ready" | "error";

export function ForgettingCurve() {
  const { buildPayload, selected, zoteroReady } = useCorpusSource();
  const [state, setState] = useState<RunState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ForgettingCurveResponse | null>(null);
  const [nComponents, setNComponents] = useState<number>(5);
  const [dropFraction, setDropFraction] = useState<number>(0.2);
  const [nIterations, setNIterations] = useState<number>(20);

  const canRun =
    state !== "loading" && (selected.kind === "hardcoded" || zoteroReady);

  const run = useCallback(async () => {
    setState("loading");
    setError(null);
    try {
      const result = await backendPost<ForgettingCurveResponse>(
        "/forgetting-curve",
        {
          corpus: buildPayload(),
          n_components: nComponents,
          drop_fraction: dropFraction,
          n_iterations: nIterations,
          seed: 0,
        },
      );
      setData(result);
      setState("ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setState("error");
    }
  }, [buildPayload, nComponents, dropFraction, nIterations]);

  return (
    <section className="space-y-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h2 className="font-display text-xl text-ink">Forgetting Curve</h2>
          <p className="text-sm text-ink/70 mt-1 max-w-3xl">
            Bootstrap over the corpus. At each iteration, remove a
            random fraction of documents, recompute the eigenbasis, and
            align it to the baseline. Eigendirections with a tight
            distribution across iterations are robust; directions whose
            agreement scatters widely are fragile under corpus
            resampling. A formal stability gate for every other finding
            Theoryscope produces.
          </p>
        </div>
        <ExportButton payload={data} filename="theoryscope-forgetting" />
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 border border-ink/10 bg-ivory/40">
        <NumberControl
          label="Components"
          value={nComponents}
          min={2}
          max={10}
          onChange={setNComponents}
        />
        <NumberControl
          label="Drop fraction"
          value={dropFraction}
          min={0.05}
          max={0.5}
          step={0.05}
          onChange={setDropFraction}
          format={(v) => `${(v * 100).toFixed(0)}%`}
        />
        <NumberControl
          label="Iterations"
          value={nIterations}
          min={5}
          max={50}
          step={1}
          onChange={setNIterations}
        />
      </div>

      <div className="flex">
        <button
          type="button"
          onClick={() => void run()}
          disabled={!canRun}
          className="ml-auto px-3 py-1.5 bg-burgundy text-ivory text-sm tracking-wide uppercase disabled:opacity-50 hover:bg-burgundy-700 transition-colors"
        >
          {state === "loading" ? "Bootstrapping…" : "Run bootstrap"}
        </button>
      </div>

      {state === "loading" ? (
        <div className="p-8 border border-ink/10 bg-ivory/50 text-ink/70">
          Running {nIterations} bootstrap iterations…
        </div>
      ) : null}

      {state === "error" ? (
        <div className="p-6 border border-rose-300 bg-rose-50/40 text-rose-800 text-sm">
          {error ?? "Unknown error."}
        </div>
      ) : null}

      {data ? <ForgettingResults data={data} /> : null}
    </section>
  );
}

function ForgettingResults({ data }: { data: ForgettingCurveResponse }) {
  return (
    <div className="space-y-6">
      <div className="border border-ink/10 bg-white/60 p-4 space-y-3">
        <div className="flex items-baseline justify-between">
          <h3 className="font-display text-base text-ink">Overall stability</h3>
          <span className="text-xs font-mono text-ink/50">
            mean stability across {data.n_iterations} bootstraps · drop{" "}
            {(data.drop_fraction * 100).toFixed(0)}%
          </span>
        </div>
        <StabilityBar value={data.overall_stability} label="stability" />
      </div>

      <div>
        <h3 className="font-display text-base text-ink mb-3">
          Per-component distribution
        </h3>
        <div className="space-y-3">
          {data.per_pc_mean.map((mean, pc) => (
            <PerPCCard
              key={pc}
              pc={pc}
              mean={mean}
              std={data.per_pc_std[pc] ?? 0}
              min={data.per_pc_min[pc] ?? 0}
              p25={data.per_pc_p25[pc] ?? 0}
              p75={data.per_pc_p75[pc] ?? 0}
            />
          ))}
        </div>
      </div>

      <details className="border border-ink/10 bg-ivory/40 p-4">
        <summary className="cursor-pointer text-sm font-semibold text-ink">
          Deep dive · per-iteration table
        </summary>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-left bg-ivory/60 text-ink/70">
                <th className="p-2 border-b border-ink/10">Iter</th>
                {data.per_pc_mean.map((_, pc) => (
                  <th
                    key={`h-${pc}`}
                    className="text-right p-2 border-b border-ink/10"
                  >
                    PC{pc + 1}
                  </th>
                ))}
                <th className="text-right p-2 border-b border-ink/10">
                  overall
                </th>
              </tr>
            </thead>
            <tbody>
              {data.per_iteration.map((row, i) => (
                <tr key={`row-${i}`} className="border-b border-ink/5">
                  <td className="p-2 text-ink/70">#{i + 1}</td>
                  {row.map((v, pc) => (
                    <td
                      key={`c-${i}-${pc}`}
                      className="text-right p-2 text-ink/70 tabular-nums"
                    >
                      {v.toFixed(3)}
                    </td>
                  ))}
                  <td className="text-right p-2 text-ink/70 tabular-nums">
                    {(data.per_iteration_stability[i] ?? 0).toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <pre className="mt-4 overflow-x-auto text-xs font-mono text-ink/80">
{JSON.stringify(data.provenance, null, 2)}
        </pre>
      </details>
    </div>
  );
}

function PerPCCard({
  pc,
  mean,
  std,
  min,
  p25,
  p75,
}: {
  pc: number;
  mean: number;
  std: number;
  min: number;
  p25: number;
  p75: number;
}) {
  return (
    <div className="border border-ink/10 bg-white/50 p-3 space-y-2">
      <div className="flex items-baseline justify-between">
        <span className="font-display text-sm text-ink">PC{pc + 1}</span>
        <span className="text-xs font-mono text-ink/60">
          mean {mean.toFixed(3)} · σ {std.toFixed(3)} · min {min.toFixed(3)}
        </span>
      </div>
      <div className="relative h-3 bg-ink/10 overflow-hidden">
        <div
          className="absolute h-full bg-amber-600/70"
          style={{
            left: `${Math.max(0, p25) * 100}%`,
            width: `${Math.max(0, p75 - p25) * 100}%`,
          }}
        />
        <div
          className="absolute top-0 h-full w-0.5 bg-emerald-800"
          style={{ left: `${Math.max(0, Math.min(1, mean)) * 100}%` }}
        />
        <div
          className="absolute top-0 h-full w-0.5 bg-rose-800/70"
          style={{ left: `${Math.max(0, Math.min(1, min)) * 100}%` }}
        />
      </div>
      <div className="flex justify-between text-[10px] font-mono text-ink/50">
        <span>0</span>
        <span>IQR amber · mean green · min rose</span>
        <span>1</span>
      </div>
    </div>
  );
}

function NumberControl({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <label className="flex flex-col text-xs text-ink/70 gap-1">
      <span className="uppercase tracking-widest text-ink/50">{label}</span>
      <div className="flex items-center gap-2">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-40 accent-burgundy"
        />
        <span className="font-mono text-ink tabular-nums w-14 text-right">
          {format ? format(value) : value}
        </span>
      </div>
    </label>
  );
}
