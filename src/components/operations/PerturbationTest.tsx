"use client";

import { useCallback, useMemo, useState } from "react";
import { backendPost } from "@/hooks/useBackend";
import { ExportButton } from "@/components/shared/ExportButton";
import { StabilityBar } from "@/components/viz/StabilityBar";
import { useCorpusSource } from "@/context/CorpusSourceContext";
import type { PerturbationResponse } from "@/types/critique";

type RunState = "idle" | "loading" | "ready" | "error";

const DEFAULT_PROBES: { label: string; text: string }[] = [
  {
    label: "Commercial press release",
    text: "Our new enterprise AI platform delivers unprecedented productivity gains across marketing, sales, and finance, helping organisations transform digital operations, accelerate time to value, and drive shareholder returns through cutting-edge generative AI capabilities.",
  },
  {
    label: "Medieval canon law",
    text: "The ecclesiastical canon holds that the bishop, by virtue of his consecrated office, exercises ordinary jurisdiction over the clergy and faithful of his diocese, subject to the metropolitan see and the universal authority of the Roman pontiff, in matters of discipline, doctrine, and sacramental order.",
  },
  {
    label: "Nature writing",
    text: "The river turned brown after the storm, carrying leaves and branches in a slow procession past the beech wood. A heron lifted from the reeds at my approach, its wings a long soft beat against the evening light, and I watched it cross the field to settle on a post.",
  },
];

export function PerturbationTest() {
  const { buildPayload, selected, zoteroReady } = useCorpusSource();
  const [state, setState] = useState<RunState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<PerturbationResponse | null>(null);
  const [probeText, setProbeText] = useState<string>(DEFAULT_PROBES[0].text);
  const [probeLabel, setProbeLabel] = useState<string>(DEFAULT_PROBES[0].label);
  const [nComponents, setNComponents] = useState<number>(5);

  const canRun =
    state !== "loading" &&
    probeText.trim().length > 0 &&
    (selected.kind === "hardcoded" || zoteroReady);

  const run = useCallback(async () => {
    setState("loading");
    setError(null);
    try {
      const result = await backendPost<PerturbationResponse>(
        "/perturbation-test",
        {
          corpus: buildPayload(),
          perturbation_text: probeText,
          perturbation_label: probeLabel || "perturbation",
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
  }, [buildPayload, probeText, probeLabel, nComponents]);

  const topRotated = useMemo(() => {
    if (!data) return [];
    const rot = data.alignment.per_component_rotation ?? [];
    return rot
      .map((r, i) => ({ pc: i, rotation: r }))
      .sort((a, b) => b.rotation - a.rotation);
  }, [data]);

  return (
    <section className="space-y-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h2 className="font-display text-xl text-ink">Perturbation Test</h2>
          <p className="text-sm text-ink/70 mt-1 max-w-3xl">
            Add one out-of-field text to the corpus, recompute the
            eigenbasis, and report how far each principal component
            rotated. Components that rotate substantially under one new
            paper were not structural features of the field. Components
            that do not rotate are more robust findings.
          </p>
        </div>
        <ExportButton payload={data} filename="theoryscope-perturbation" />
      </header>

      <div className="space-y-3 p-4 border border-ink/10 bg-ivory/40">
        <div className="flex flex-wrap gap-2">
          {DEFAULT_PROBES.map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() => {
                setProbeText(p.text);
                setProbeLabel(p.label);
              }}
              className={[
                "px-2.5 py-1 text-xs uppercase tracking-wide border transition-colors",
                probeLabel === p.label
                  ? "border-burgundy text-ink bg-white/70"
                  : "border-ink/20 text-ink/70 bg-white/40 hover:bg-white/60",
              ].join(" ")}
            >
              {p.label}
            </button>
          ))}
        </div>
        <label className="flex flex-col text-xs text-ink/70 gap-1">
          <span className="uppercase tracking-widest text-ink/50">Label</span>
          <input
            type="text"
            value={probeLabel}
            onChange={(e) => setProbeLabel(e.target.value)}
            className="px-2 py-1.5 bg-white/60 border border-ink/20 text-sm font-mono"
            placeholder="short label for provenance"
          />
        </label>
        <label className="flex flex-col text-xs text-ink/70 gap-1">
          <span className="uppercase tracking-widest text-ink/50">
            Out-of-field text
          </span>
          <textarea
            value={probeText}
            onChange={(e) => setProbeText(e.target.value)}
            rows={6}
            className="px-2 py-1.5 bg-white/60 border border-ink/20 text-sm"
            placeholder="Paste one paper abstract, passage, or press release…"
          />
        </label>
        <div className="flex items-end justify-between gap-6">
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
            className="px-3 py-1.5 bg-burgundy text-ivory text-sm tracking-wide uppercase disabled:opacity-50 hover:bg-burgundy-700 transition-colors"
          >
            {state === "loading" ? "Perturbing…" : "Run perturbation"}
          </button>
        </div>
      </div>

      {state === "loading" ? (
        <div className="p-8 border border-ink/10 bg-ivory/50 text-ink/70">
          Embedding the perturbation and recomputing the eigenbasis…
        </div>
      ) : null}

      {state === "error" ? (
        <div className="p-6 border border-rose-300 bg-rose-50/40 text-rose-800 text-sm">
          {error ?? "Unknown error."}
        </div>
      ) : null}

      {data ? (
        <div className="space-y-6">
          <div className="border border-ink/10 bg-white/60 p-4 space-y-3">
            <div className="flex items-baseline justify-between">
              <h3 className="font-display text-base text-ink">
                Overall stability
              </h3>
              <span className="text-xs font-mono text-ink/50">
                mean |cos(baseline, perturbed)| across matched PCs
              </span>
            </div>
            <StabilityBar
              value={data.alignment.stability}
              label="stability"
            />
          </div>

          <div>
            <h3 className="font-display text-base text-ink mb-3">
              Ranked rotation by component
            </h3>
            <div className="space-y-3">
              {topRotated.map(({ pc, rotation }) => (
                <div
                  key={`rot-${pc}`}
                  className="border border-ink/10 bg-white/50 p-3 space-y-2"
                >
                  <div className="flex items-baseline justify-between">
                    <span className="font-display text-sm text-ink">
                      PC{pc + 1}
                    </span>
                    <span className="text-xs font-mono text-ink/60">
                      rotation = {rotation.toFixed(3)} · probe scored{" "}
                      {(data.probe.projection_on_perturbed_basis[pc] ?? 0).toFixed(
                        3,
                      )}
                    </span>
                  </div>
                  <StabilityBar
                    value={1 - rotation}
                    label="stability"
                  />
                </div>
              ))}
            </div>
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
      ) : null}
    </section>
  );
}
