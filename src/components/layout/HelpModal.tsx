"use client";

import { useCallback, useEffect } from "react";
import { VERSION } from "@/lib/version";

export function HelpModal({ onClose }: { onClose: () => void }) {
  // Close on Escape.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const stop = useCallback((e: React.MouseEvent) => e.stopPropagation(), []);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="help-modal-title"
    >
      <div
        className="bg-ivory border border-ink/20 max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col shadow-xl"
        onClick={stop}
      >
        <header className="px-6 py-4 border-b border-ink/10 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <img
              src="/icon.svg"
              alt=""
              width={32}
              height={32}
              aria-hidden="true"
            />
            <div>
              <h2
                id="help-modal-title"
                className="font-display text-xl text-ink leading-tight"
              >
                Theoryscope
              </h2>
              <p className="text-xs text-ink/60 tracking-wide uppercase mt-0.5">
                Geometry of Theory Space · v{VERSION}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-ink/60 hover:text-ink text-lg leading-none px-2"
            aria-label="Close"
          >
            ✕
          </button>
        </header>

        <div className="px-6 py-5 overflow-y-auto space-y-6">
          <section className="space-y-2">
            <p className="text-sm text-ink/70 leading-relaxed max-w-prose">
              Theoryscope treats a corpus of theoretical texts as a geometry
              and asks renormalisation-group and eigenvector questions about
              it. Load a corpus (Phase 0 ships with 20 philosophy-of-technology
              texts; a Zotero collection can be pointed at instead), then run
              operations across three groups — <strong>Inspect</strong> for
              principal axes and the cloud itself, <strong>Flow</strong> for
              coarse-graining and its fixed points, <strong>Critique</strong>{" "}
              for stability diagnostics that tell you whether a finding should
              be trusted.
            </p>
            <p className="text-sm text-ink/70 leading-relaxed max-w-prose">
              Every operation writes a provenance record (corpus hash,
              embedding model, operator, chunking spec) into its export so
              that a Theoryscope finding can travel into a paper.
            </p>
          </section>

          <section className="space-y-2">
            <h3 className="text-[11px] uppercase tracking-widest text-ink/50">
              Inspect
            </h3>
            <ul className="text-sm text-ink/80 space-y-1.5">
              <li>
                <strong>Corpus Map.</strong> 3D PCA scatter of the cloud.
              </li>
              <li>
                <strong>Eigendirections.</strong> Annotated principal axes,
                positive and negative poles per component.
              </li>
            </ul>
          </section>

          <section className="space-y-2">
            <h3 className="text-[11px] uppercase tracking-widest text-ink/50">
              Flow
            </h3>
            <ul className="text-sm text-ink/80 space-y-1.5">
              <li>
                <strong>Coarse-Graining Trajectory.</strong> Aggregative
                k-means flow with a Play/Pause scrubber and a schedule bar.
              </li>
              <li>
                <strong>Fixed Point Finder.</strong> Terminal basins with
                exemplars; click a basin to highlight its members.
              </li>
              <li>
                <strong>Universality Classes.</strong> Classes ranked by
                ascending surface cosine — low cosine is the universality
                finding to read.
              </li>
            </ul>
          </section>

          <section className="space-y-2">
            <h3 className="text-[11px] uppercase tracking-widest text-ink/50">
              Critique
            </h3>
            <ul className="text-sm text-ink/80 space-y-1.5">
              <li>
                <strong>Embedding Dependence Probe.</strong> Re-run under a
                second open-weight model; findings that survive are stronger
                claims about the field.
              </li>
              <li>
                <strong>Perturbation Test.</strong> Add one out-of-field text
                and measure per-component rotation.
              </li>
              <li>
                <strong>Forgetting Curve.</strong> Bootstrap over the corpus;
                report per-component stability as mean, IQR, and minimum.
              </li>
            </ul>
          </section>

          <section className="space-y-2">
            <h3 className="text-[11px] uppercase tracking-widest text-ink/50">
              Methodological commitments
            </h3>
            <ul className="text-sm text-ink/80 space-y-1.5 max-w-prose">
              <li>
                Open-weight embedding models only. Commercial APIs hide their
                tokeniser, architecture, training data, and update schedule,
                and their outputs cannot be re-run.
              </li>
              <li>
                Every operator implemented in-process, with its parameters
                visible on screen.
              </li>
              <li>
                Stability under re-embedding is part of every result, not a
                bonus.
              </li>
              <li>
                Corpus provenance travels with every export.
              </li>
              <li>
                The tool produces geometry; the critical reading is the
                critic&apos;s work.
              </li>
            </ul>
          </section>

          <section className="space-y-2">
            <h3 className="text-[11px] uppercase tracking-widest text-ink/50">
              Keyboard
            </h3>
            <p className="text-sm text-ink/80">
              <kbd className="px-1.5 py-0.5 text-xs font-mono border border-ink/20 bg-white/60">
                Esc
              </kbd>{" "}
              closes this dialog and the Zotero credentials dialog.
            </p>
          </section>
        </div>

        <footer className="border-t border-ink/10 px-6 py-4 flex items-center justify-between gap-4 bg-ivory/60">
          <a
            href="https://vector-lab-tools.github.io"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-2 text-xs text-ink/60 hover:text-ink transition-colors"
          >
            <img
              src="/vector-lab-mark.svg"
              alt=""
              width={24}
              height={24}
              aria-hidden="true"
            />
            <span>
              Part of the{" "}
              <span className="underline decoration-ink/20 group-hover:decoration-ink">
                Vector Lab
              </span>
            </span>
          </a>
          <span className="text-[10px] font-mono text-ink/40">
            vector-lab-tools / theoryscope
          </span>
        </footer>
      </div>
    </div>
  );
}
