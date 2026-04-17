"use client";

import { useState } from "react";
import { CoarseGrainingTrajectory } from "@/components/operations/CoarseGrainingTrajectory";
import { CorpusMap } from "@/components/operations/CorpusMap";
import { Eigendirections } from "@/components/operations/Eigendirections";
import { FixedPointFinder } from "@/components/operations/FixedPointFinder";
import { UniversalityClasses } from "@/components/operations/UniversalityClasses";
import { CorpusLoader } from "@/components/layout/CorpusLoader";
import { CorpusSourcePanel } from "@/components/layout/CorpusSourcePanel";
import { Header } from "@/components/layout/Header";
import { StatusBar } from "@/components/layout/StatusBar";
import Providers from "./providers";

type Tab = {
  key: string;
  label: string;
  group: "Inspect" | "Flow" | "Critique" | "Annotations" | "Atlas";
  component: React.ReactNode | null;
};

const TABS: Tab[] = [
  { key: "corpus-map", label: "Corpus Map", group: "Inspect", component: <CorpusMap /> },
  { key: "eigendirections", label: "Eigendirections", group: "Inspect", component: <Eigendirections /> },
  { key: "concept-locator", label: "Concept Locator", group: "Inspect", component: null },
  { key: "author-constellation", label: "Author Constellation", group: "Inspect", component: null },
  { key: "coarse-graining", label: "Coarse-Graining Trajectory", group: "Flow", component: <CoarseGrainingTrajectory /> },
  { key: "fixed-points", label: "Fixed Point Finder", group: "Flow", component: <FixedPointFinder /> },
  { key: "operator-spectrum", label: "Relevant / Irrelevant Operators", group: "Flow", component: null },
  { key: "universality", label: "Universality Classes", group: "Flow", component: <UniversalityClasses /> },
  { key: "temporal-flow", label: "Temporal RG Flow", group: "Flow", component: null },
  { key: "symmetry-breaking", label: "Symmetry Breaking Map", group: "Critique", component: null },
  { key: "phase-diagram", label: "Phase Diagram", group: "Critique", component: null },
  { key: "embedding-probe", label: "Embedding Dependence Probe", group: "Critique", component: null },
  { key: "perturbation", label: "Perturbation Test", group: "Critique", component: null },
  { key: "forgetting", label: "Forgetting Curve", group: "Critique", component: null },
  { key: "translation", label: "Translated Corpus Probe", group: "Critique", component: null },
  { key: "corpus-vs-model", label: "Corpus-vs-Model Probe", group: "Critique", component: null },
  { key: "annotations", label: "Critical Annotations", group: "Annotations", component: null },
  { key: "atlas", label: "Atlas", group: "Atlas", component: null },
];

export default function Home() {
  return (
    <Providers>
      <Shell />
    </Providers>
  );
}

function Shell() {
  const [active, setActive] = useState<string>("corpus-map");
  const currentTab = TABS.find((t) => t.key === active) ?? TABS[0];

  const groups = ["Inspect", "Flow", "Critique", "Annotations", "Atlas"] as const;

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <div className="px-6 pt-4 space-y-3">
        <CorpusSourcePanel />
        <CorpusLoader />
      </div>

      <nav className="px-6 pt-6 pb-2 flex flex-wrap gap-x-6 gap-y-2 border-b border-ink/10">
        {groups.map((group) => (
          <div key={group} className="flex items-baseline gap-3">
            <span className="text-[10px] uppercase tracking-widest text-ink/40">
              {group}
            </span>
            {TABS.filter((t) => t.group === group).map((t) => {
              const isActive = t.key === active;
              const isLive = t.component !== null;
              return (
                <button
                  key={t.key}
                  type="button"
                  onClick={() => isLive && setActive(t.key)}
                  className={[
                    "text-sm pb-1 border-b-2 transition-colors",
                    isActive
                      ? "border-burgundy text-ink"
                      : isLive
                      ? "border-transparent text-ink/70 hover:text-ink"
                      : "border-transparent text-ink/30 cursor-not-allowed",
                  ].join(" ")}
                  disabled={!isLive}
                  title={isLive ? "" : "Coming in a later phase"}
                >
                  {t.label}
                </button>
              );
            })}
          </div>
        ))}
      </nav>

      <main className="flex-1 p-6">
        {currentTab.component ?? (
          <div className="p-8 text-ink/60">
            This operation will be implemented in a later phase. See{" "}
            <code className="px-1 bg-ink/5">WORKING.md</code> for the roadmap.
          </div>
        )}
      </main>

      <StatusBar />
    </div>
  );
}
