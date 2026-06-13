"use client";

import { useState } from "react";
import { HelpModal } from "@/components/layout/HelpModal";
import { VERSION } from "@/lib/version";

type Props = {
  /** Label of the currently-active tab (displayed to the right of the tool name). */
  currentTabLabel?: string;
};

/**
 * Header composition follows the Vector Lab convention now standardised
 * across Manifold Atlas and Diffusion Atlas: dark-navy family mark
 * (#0a1f44 VL polygon) + VECTOR LAB wordmark | per-tool branded SVG +
 * tool name | current mode label. Theoryscope's tool icon stays gold
 * (#b8941e — inner tier, single-corpus scope).
 *
 * Asset paths use the canonical `public/icons/vector-lab-*.svg` naming.
 */
export function Header({ currentTabLabel }: Props) {
  const [helpOpen, setHelpOpen] = useState(false);

  return (
    <>
      <header className="border-b border-ink/10 bg-ivory/80 backdrop-blur px-4 py-2 flex items-center gap-3">
        {/* Vector Lab family mark */}
        <a
          href="https://vector-lab-tools.github.io"
          target="_blank"
          rel="noopener noreferrer"
          title="Vector Lab — research instruments for critical vector theory"
          className="flex items-center gap-1.5 opacity-80 hover:opacity-100 transition-opacity shrink-0"
        >
          <img
            src="/icons/vector-lab-logo-mark.svg"
            alt="Vector Lab"
            width={20}
            height={20}
            className="w-5 h-5"
          />
          <span className="text-[10px] font-medium tracking-widest uppercase text-ink/60 hidden sm:inline">
            Vector Lab
          </span>
        </a>

        <div className="h-4 w-px bg-parchment" />

        {/* Theoryscope identity */}
        <div className="flex items-center gap-2 shrink-0">
          <img
            src="/icons/vector-lab-theoryscope.svg"
            alt=""
            width={20}
            height={20}
            className="w-5 h-5"
            aria-hidden="true"
          />
          <h1 className="font-display text-sm font-bold text-ink leading-none">
            Theoryscope
          </h1>
        </div>

        {currentTabLabel ? (
          <>
            <div className="h-4 w-px bg-parchment" />
            <span className="text-xs text-ink/60 truncate">
              {currentTabLabel}
            </span>
          </>
        ) : null}

        <div className="flex-1" />

        <div className="flex items-center gap-3 text-xs text-ink/50 shrink-0">
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
