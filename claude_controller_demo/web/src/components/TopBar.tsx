import { useLab, type Tab } from "../store/lab";
import { cx } from "./ui";

const TABS: { id: Tab; label: string; sub: string }[] = [
  { id: "live", label: "Live Episode", sub: "step through the loop" },
  { id: "bench", label: "Benchmark Runner", sub: "many rollouts" },
  { id: "data", label: "Trajectories → BC", sub: "build a dataset" },
  { id: "learn", label: "Concepts", sub: "how it maps to RL" },
];

export function TopBar() {
  const tab = useLab((s) => s.tab);
  const setTab = useLab((s) => s.setTab);
  const connection = useLab((s) => s.connection);
  const config = useLab((s) => s.config);

  return (
    <header className="flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-ink-700 bg-ink-950/80 px-4 py-2 backdrop-blur">
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-sm font-bold text-policy">π</span>
        <h1 className="text-sm font-semibold tracking-tight">
          Claude as RL Controller <span className="text-ink-500">· BrowserGym</span>
        </h1>
      </div>
      <nav className="flex gap-1">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={cx(
              "rounded-md px-3 py-1.5 text-left transition",
              tab === t.id ? "bg-ink-800 text-ink-100" : "text-ink-500 hover:text-ink-100",
            )}
          >
            <div className="text-sm font-medium leading-tight">{t.label}</div>
            <div className="text-[10px] leading-tight text-ink-500">{t.sub}</div>
          </button>
        ))}
      </nav>
      <div className="ml-auto flex items-center gap-4 font-mono text-[11px]">
        <Status ok={connection === "open"} label={`bridge ${connection}`} />
        <Status
          ok={!!config?.has_api_key}
          label={config ? (config.has_api_key ? "ANTHROPIC_API_KEY set" : "no API key: random/human only") : "…"}
        />
      </div>
    </header>
  );
}

function Status({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className="flex items-center gap-1.5 text-ink-300">
      <span className={cx("h-1.5 w-1.5 rounded-full", ok ? "bg-reward" : "bg-act")} />
      {label}
    </span>
  );
}
