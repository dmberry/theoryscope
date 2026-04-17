"use client";

/**
 * A horizontal bar showing a stability score (0–1). Colour shifts from
 * rose (unstable) through amber to emerald (stable) so results are
 * legible at a glance without reading the number.
 */
export function StabilityBar({
  value,
  label,
  width = "w-full",
}: {
  value: number;
  label?: string;
  width?: string;
}) {
  const v = Math.max(0, Math.min(1, value));
  const pct = v * 100;
  const color =
    v >= 0.8
      ? "bg-emerald-700"
      : v >= 0.5
      ? "bg-amber-600"
      : "bg-rose-700";
  return (
    <div className={`${width} flex items-center gap-3`}>
      {label ? (
        <span className="text-xs uppercase tracking-widest text-ink/50 shrink-0">
          {label}
        </span>
      ) : null}
      <div className="flex-1 h-1.5 bg-ink/10 overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-ink/70 tabular-nums shrink-0 w-12 text-right">
        {v.toFixed(3)}
      </span>
    </div>
  );
}
