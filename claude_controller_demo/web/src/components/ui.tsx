import type { ButtonHTMLAttributes, ReactNode } from "react";

export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");

export function Panel({
  title,
  accent,
  right,
  children,
  className,
  bodyClass,
}: {
  title: ReactNode;
  accent?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
}) {
  return (
    <section
      className={cx(
        "flex min-h-0 flex-col rounded-lg border border-ink-700 bg-ink-900/90 shadow-lg shadow-black/30",
        className,
      )}
    >
      <header className="flex items-center gap-2 border-b border-ink-700 px-3 py-2">
        {accent && <span className="h-2 w-2 rounded-full" style={{ background: accent }} />}
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-300">{title}</h2>
        <div className="ml-auto flex items-center gap-2">{right}</div>
      </header>
      <div className={cx("min-h-0 flex-1", bodyClass)}>{children}</div>
    </section>
  );
}

type Variant = "primary" | "ghost" | "danger" | "policy";
export function Button({
  variant = "ghost",
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  const styles: Record<Variant, string> = {
    primary: "bg-ink-100 text-ink-950 hover:bg-white",
    policy: "bg-policy text-ink-950 hover:brightness-110",
    danger: "bg-act/90 text-ink-950 hover:bg-act",
    ghost: "border border-ink-700 bg-ink-850 text-ink-100 hover:border-ink-500",
  };
  return (
    <button
      {...props}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition",
        "disabled:cursor-not-allowed disabled:opacity-40",
        styles[variant],
        className,
      )}
    />
  );
}

export function Badge({ children, color }: { children: ReactNode; color?: string }) {
  return (
    <span
      className="inline-flex items-center rounded px-1.5 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wide"
      style={{
        color: color ?? "var(--color-ink-300)",
        background: `color-mix(in srgb, ${color ?? "var(--color-ink-500)"} 16%, transparent)`,
      }}
    >
      {children}
    </span>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="flex flex-col gap-1" title={hint}>
      <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-500">{label}</span>
      {children}
    </label>
  );
}

export const inputCls =
  "rounded-md border border-ink-700 bg-ink-950 px-2 py-1.5 text-sm text-ink-100 outline-none focus:border-policy";

export function Toggle({ checked, onChange, label, hint }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) {
  return (
    <button
      type="button"
      title={hint}
      onClick={() => onChange(!checked)}
      className="flex items-center gap-2 text-xs text-ink-300 hover:text-ink-100"
    >
      <span className={cx("relative h-4 w-7 rounded-full transition", checked ? "bg-policy" : "bg-ink-700")}>
        <span className={cx("absolute top-0.5 h-3 w-3 rounded-full bg-ink-950 transition-all", checked ? "left-3.5" : "left-0.5")} />
      </span>
      {label}
    </button>
  );
}

export const sourceColor = (s: string) =>
  s === "claude" ? "var(--color-policy)" : s === "human" ? "var(--color-act)" : "var(--color-ink-300)";
