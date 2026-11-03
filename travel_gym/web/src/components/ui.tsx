import type { ButtonHTMLAttributes, ReactNode } from "react";
import type { PageType } from "../shared/schemas";

export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");

export const PAGE_COLOR: Record<PageType, string> = {
  search: "var(--color-pg-search)",
  results: "var(--color-pg-results)",
  details: "var(--color-pg-details)",
  review: "var(--color-pg-review)",
  confirmation: "var(--color-pg-confirmation)",
  other: "var(--color-pg-other)",
};

export function Card({ title, right, children, className }: {
  title?: ReactNode; right?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={cx("rounded-xl border border-slate-200 bg-white shadow-sm", className)}>
      {title && (
        <header className="flex items-center gap-2 border-b border-slate-100 px-4 py-2.5">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">{title}</h2>
          <div className="ml-auto">{right}</div>
        </header>
      )}
      {children}
    </section>
  );
}

export function Button({ variant = "default", className, ...p }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "default" | "danger" }) {
  const v = {
    primary: "bg-sky-600 text-white hover:bg-sky-700",
    default: "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50",
    danger: "bg-rose-600 text-white hover:bg-rose-700",
  }[variant];
  return <button {...p} className={cx("rounded-lg px-3 py-1.5 text-sm font-medium transition disabled:opacity-40", v, className)} />;
}

type Tone = "slate" | "green" | "red" | "amber" | "sky" | "violet";
export function Pill({ children, tone = "slate" }: { children: ReactNode; tone?: Tone }) {
  const t: Record<Tone, string> = {
    slate: "bg-slate-100 text-slate-600",
    green: "bg-emerald-50 text-emerald-700",
    red: "bg-rose-50 text-rose-700",
    amber: "bg-amber-50 text-amber-700",
    sky: "bg-sky-50 text-sky-700",
    violet: "bg-violet-50 text-violet-700",
  };
  return <span className={cx("inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium", t[tone])}>{children}</span>;
}

export const money = (x: number | null | undefined) =>
  x == null ? "–" : `$${x.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div>
      <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">{label}</div>
      <div className="text-lg font-semibold tabular-nums">{value}</div>
      {sub && <div className="text-xs text-slate-500">{sub}</div>}
    </div>
  );
}
