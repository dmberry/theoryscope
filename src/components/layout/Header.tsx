"use client";

import { useState } from "react";
import { HelpModal } from "@/components/layout/HelpModal";
import { VERSION } from "@/lib/version";

export function Header() {
  const [helpOpen, setHelpOpen] = useState(false);

  return (
    <>
      <header className="border-b border-ink/10 bg-ivory/80 backdrop-blur px-6 py-4 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          {/* Tool mark: point cloud with eigenvector axes. Vector Lab inner-tier gold. */}
          <img
            src="/icon.svg"
            alt=""
            width={32}
            height={32}
            className="shrink-0"
            aria-hidden="true"
          />
          <div>
            <h1 className="font-display text-2xl text-ink leading-tight">
              Theoryscope
            </h1>
            <p className="text-xs text-ink/60 mt-0.5 tracking-wide uppercase">
              Geometry of Theory Space
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs text-ink/50">
          <a
            href="https://vector-lab-tools.github.io"
            target="_blank"
            rel="noreferrer"
            className="hover:text-ink underline decoration-ink/20 hover:decoration-ink"
            title="Part of the Vector Lab"
          >
            Part of the Vector Lab
          </a>
          <button
            type="button"
            onClick={() => setHelpOpen(true)}
            className="px-2 py-1 border border-ink/20 bg-white/60 text-ink/70 hover:text-ink hover:bg-ivory/60 uppercase tracking-wide transition-colors"
            aria-label="Open help and about"
          >
            Help
          </button>
          <span className="font-mono">v{VERSION}</span>
        </div>
      </header>

      {helpOpen ? <HelpModal onClose={() => setHelpOpen(false)} /> : null}
    </>
  );
}
