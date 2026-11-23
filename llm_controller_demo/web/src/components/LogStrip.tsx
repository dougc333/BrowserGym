import { useLab } from "../store/lab";

export function LogStrip() {
  const log = useLab((s) => s.log);
  const last = log.at(-1);
  if (!last) return null;
  return (
    <footer className="flex items-center gap-3 border-t border-ink-700 bg-ink-950 px-4 py-1.5 font-mono text-[11px]">
      <span className={last.level === "error" ? "text-act" : "text-ink-500"}>
        {last.level === "error" ? "error" : "info"}
      </span>
      <span className="truncate text-ink-300">{last.message}</span>
      <span className="ml-auto shrink-0 text-ink-500">{new Date(last.ts).toLocaleTimeString()}</span>
    </footer>
  );
}
