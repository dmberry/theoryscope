"use client";

import { useMemo } from "react";
import type { CorpusDocument } from "@/types/corpus";

type Point = {
  x: number;
  y: number;
  label: number;
  doc: CorpusDocument;
};

type Props = {
  coords: [number, number][];
  labels: number[];
  documents: CorpusDocument[];
  /** Optional: document indices to highlight (dim all others). */
  highlightIndices?: number[];
  /** Optional: fixed palette size (for stable colours across steps). */
  paletteSize?: number;
  /** Optional: height in pixels. Defaults to 420. */
  height?: number;
  onDocClick?: (docIndex: number) => void;
};

/**
 * Lightweight SVG scatter for the Flow operations. Uses a deterministic
 * categorical palette so that cluster colours stay stable across steps
 * and across operations.
 */
export function FlowScatter({
  coords,
  labels,
  documents,
  highlightIndices,
  paletteSize,
  height = 420,
  onDocClick,
}: Props) {
  const n = Math.min(coords.length, labels.length, documents.length);
  const points = useMemo<Point[]>(
    () =>
      Array.from({ length: n }, (_, i) => ({
        x: coords[i][0],
        y: coords[i][1],
        label: labels[i],
        doc: documents[i],
      })),
    [coords, labels, documents, n],
  );

  const bounds = useMemo(() => computeBounds(points), [points]);
  const maxLabel = useMemo(() => {
    if (paletteSize && paletteSize > 0) return paletteSize;
    return Math.max(1, ...labels) + 1;
  }, [labels, paletteSize]);

  const highlightSet = useMemo(
    () => (highlightIndices ? new Set(highlightIndices) : null),
    [highlightIndices],
  );

  const width = 720;
  const pad = 24;

  return (
    <div className="border border-ink/10 bg-white/60" style={{ height }}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        className="w-full h-full"
      >
        {points.map((p, i) => {
          const cx = projectX(p.x, bounds, width, pad);
          const cy = projectY(p.y, bounds, height, pad);
          const dim = highlightSet && !highlightSet.has(i);
          const color = colorFor(p.label, maxLabel);
          return (
            <g
              key={p.doc.id}
              onClick={onDocClick ? () => onDocClick(i) : undefined}
              className={onDocClick ? "cursor-pointer" : undefined}
            >
              <title>
                {`${p.doc.author} ${p.doc.year}\n${p.doc.title}\ncluster ${p.label}`}
              </title>
              <circle
                cx={cx}
                cy={cy}
                r={dim ? 3 : 5}
                fill={color}
                fillOpacity={dim ? 0.15 : 0.85}
                stroke="#1a1a1a"
                strokeOpacity={dim ? 0.1 : 0.5}
                strokeWidth={0.6}
              />
              <text
                x={cx + 7}
                y={cy + 3}
                fontSize={10}
                fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
                fill="#222"
                fillOpacity={dim ? 0.25 : 0.85}
              >
                {shortAuthor(p.doc.author)} {p.doc.year}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/* ------------------------------------------------------------------ */

type Bounds = { minX: number; maxX: number; minY: number; maxY: number };

function computeBounds(points: Point[]): Bounds {
  if (points.length === 0) {
    return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  }
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  if (minX === maxX) {
    minX -= 0.5;
    maxX += 0.5;
  }
  if (minY === maxY) {
    minY -= 0.5;
    maxY += 0.5;
  }
  return { minX, maxX, minY, maxY };
}

function projectX(x: number, b: Bounds, width: number, pad: number): number {
  const t = (x - b.minX) / (b.maxX - b.minX);
  return pad + t * (width - 2 * pad);
}
function projectY(y: number, b: Bounds, height: number, pad: number): number {
  const t = (y - b.minY) / (b.maxY - b.minY);
  // SVG y axis grows downward; flip so that higher y in data space is higher on screen.
  return height - pad - t * (height - 2 * pad);
}

/** Qualitative 12-stop palette (editorial-friendly, hand-picked). */
const PALETTE = [
  "#8c2f39", // burgundy
  "#c98a3b", // amber
  "#355c7d", // dusk blue
  "#497e4f", // sage
  "#6a4c93", // plum
  "#b65e4c", // terracotta
  "#3f8a97", // teal
  "#a56a3a", // tan
  "#2f6b5a", // pine
  "#855a98", // lavender
  "#a63d57", // rose
  "#4b6b9c", // slate blue
];

function colorFor(label: number, maxLabel: number): string {
  if (maxLabel <= PALETTE.length) {
    return PALETTE[label % PALETTE.length];
  }
  // Fallback HSL ring for larger cluster counts.
  const hue = Math.round((360 * label) / maxLabel);
  return `hsl(${hue}deg 50% 40%)`;
}

function shortAuthor(author: string): string {
  return author.split(",")[0] ?? author;
}
