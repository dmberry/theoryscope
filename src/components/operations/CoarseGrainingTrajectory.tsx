"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { backendPost } from "@/hooks/useBackend";
import { ExportButton } from "@/components/shared/ExportButton";
import { FlowScatter } from "@/components/viz/FlowScatter";
import { useCorpusSource } from "@/context/CorpusSourceContext";
import type { CoarseGrainingTrajectoryResponse } from "@/types/flow";

type RunState = "idle" | "loading" | "ready" | "error";

export function CoarseGrainingTrajectory() {
  const { buildPayload, selected, zoteroReady } = useCorpusSource();
  const [state, setState] = useState<RunState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<CoarseGrainingTrajectoryResponse | null>(null);
  const [nSteps, setNSteps] = useState<number>(6);
  const [stepIndex, setStepIndex] = useState<number>(0);
  const [playing, setPlaying] = useState<boolean>(false);
  const intervalRef = useRef<number | null>(null);

  const canRun =
    state !== "loading" && (selected.kind === "hardcoded" || zoteroReady);

  const run = useCallback(async () => {
    setState("loading");
    setError(null);
    setPlaying(false);
    try {
      const result = await backendPost<CoarseGrainingTrajectoryResponse>(
        "/coarse-graining-trajectory",
        {
          corpus: buildPayload(),
          n_steps: nSteps,
          seed: 0,
        },
      );
      setData(result);
      setStepIndex(0);
      setState("ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setState("error");
    }
  }, [buildPayload, nSteps]);

  // Autoplay: cycle forward through the steps when `playing` is true.
  useEffect(() => {
    if (!playing || !data) return;
    intervalRef.current = window.setInterval(() => {
      setStepIndex((i) => {
        const n = data.steps.length;
        if (n === 0) return 0;
        const next = i + 1;
        if (next >= n) {
          setPlaying(false);
          return n - 1;
        }
        return next;
      });
    }, 900);
    return () => {
      if (intervalRef.current !== null) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [playing, data]);

  const currentStep = data?.steps[stepIndex];
  const maxLabel = useMemo(() => {
    if (!data) return 1;
    return Math.max(...data.schedule) + 1;
  }, [data]);

  return (
    <section className="space-y-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h2 className="font-display text-xl text-ink">
            Coarse-Graining Trajectory
          </h2>
          <p className="text-sm text-ink/70 mt-1 max-w-3xl">
            Watch the corpus cloud collapse under progressive aggregative
            k-means coarse-graining. At each step every document is placed
            at the centroid of its current cluster; the number of clusters
            shrinks step by step so that fine-grained distinctions are
            integrated out. The operator is visible: the schedule of
            cluster counts is shown on the timeline.
          </p>
        </div>
        <ExportButton
          payload={data}
          filename="theoryscope-coarse-graining"
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
          {state === "loading" ? "Computing…" : "Compute flow"}
        </button>
      </div>

      {state === "loading" ? (
        <div className="p-8 border border-ink/10 bg-ivory/50 text-ink/70">
          Running the aggregative k-means flow across {nSteps} steps…
        </div>
      ) : null}

      {state === "error" ? (
        <div className="p-6 border border-rose-300 bg-rose-50/40 text-rose-800 text-sm">
          {error ?? "Unknown error."}
        </div>
      ) : null}

      {data && currentStep ? (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => setPlaying((p) => !p)}
              className="px-3 py-1.5 text-xs uppercase tracking-wide border border-ink/20 bg-white/60 hover:bg-ivory/60 text-ink"
            >
              {playing ? "Pause" : "Play"}
            </button>
            <input
              type="range"
              min={0}
              max={data.steps.length - 1}
              step={1}
              value={stepIndex}
              onChange={(e) => {
                setStepIndex(Number(e.target.value));
                setPlaying(false);
              }}
              className="flex-1 accent-burgundy"
            />
            <span className="text-xs font-mono text-ink/70 tabular-nums w-28 text-right">
              step {stepIndex + 1}/{data.steps.length} · k={currentStep.k}
            </span>
          </div>

          <FlowScatter
            coords={currentStep.doc_coords_2d}
            labels={currentStep.labels}
            documents={data.documents}
            paletteSize={maxLabel}
            height={460}
          />

          <ScheduleBar
            schedule={data.schedule}
            currentIndex={stepIndex}
            onPick={(i) => {
              setStepIndex(i);
              setPlaying(false);
            }}
          />

          <details className="border border-ink/10 bg-ivory/40 p-4">
            <summary className="cursor-pointer text-sm font-semibold text-ink">
              Deep dive · schedule + provenance
            </summary>
            <div className="mt-4 space-y-4 text-xs text-ink/80">
              <p className="font-mono">
                Cluster schedule: {data.schedule.join(" → ")}
              </p>
              <p className="font-mono">
                2D variance explained:{" "}
                {(data.pca2d_variance[0] * 100).toFixed(1)}% · {" "}
                {(data.pca2d_variance[1] * 100).toFixed(1)}%
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

function ScheduleBar({
  schedule,
  currentIndex,
  onPick,
}: {
  schedule: number[];
  currentIndex: number;
  onPick: (i: number) => void;
}) {
  return (
    <div className="flex items-stretch border border-ink/10">
      {schedule.map((k, i) => {
        const active = i === currentIndex;
        return (
          <button
            key={i}
            type="button"
            onClick={() => onPick(i)}
            className={[
              "flex-1 px-2 py-1.5 text-xs tabular-nums font-mono transition-colors",
              active
                ? "bg-burgundy text-ivory"
                : "bg-white/60 text-ink/70 hover:bg-ivory/60",
              i > 0 ? "border-l border-ink/10" : "",
            ].join(" ")}
          >
            k={k}
          </button>
        );
      })}
    </div>
  );
}
