import { useGym, useSelectedRun, type Tab } from "./store/gym";
import { Sidebar } from "./components/Sidebar";
import { RunHeader } from "./components/RunHeader";
import { TrajectoryView } from "./components/TrajectoryView";
import { CoverageView } from "./components/CoverageView";
import { FaresView } from "./components/FaresView";
import { cx } from "./components/ui";

const TABS: { id: Tab; label: string }[] = [
  { id: "trajectory", label: "Trajectory" },
  { id: "coverage", label: "Search coverage" },
  { id: "fares", label: "Fares vs truth" },
];

function TopBar() {
  const connected = useGym((s) => s.connected);
  const config = useGym((s) => s.config);
  return (
    <header className="flex items-center gap-4 border-b border-slate-200 bg-white px-5 py-2.5">
      <h1 className="text-sm font-semibold">Travel Search Gym</h1>
      <span className="text-xs text-slate-500">agentic search over a sandbox flight site, checked against the database</span>
      <div className="ml-auto flex items-center gap-4 text-xs text-slate-600">
        <span className="flex items-center gap-1.5">
          <span className={cx("h-2 w-2 rounded-full", connected ? "bg-emerald-500" : "bg-rose-500")} />
          {connected ? "live" : "reconnecting"}
        </span>
        <span className="flex items-center gap-1.5">
          <span className={cx("h-2 w-2 rounded-full", config?.site_up ? "bg-emerald-500" : "bg-rose-500")} />
          sandbox site
        </span>
        <span className="flex items-center gap-1.5">
          <span className={cx("h-2 w-2 rounded-full", config?.has_api_key ? "bg-emerald-500" : "bg-slate-300")} />
          {config?.has_api_key ? "Claude available" : "no API key: scripted / OpenAI-compatible only"}
        </span>
      </div>
    </header>
  );
}

export default function App() {
  const run = useSelectedRun();
  const tab = useGym((s) => s.tab);
  const setTab = useGym((s) => s.setTab);
  const error = useGym((s) => s.error);

  return (
    <div className="flex h-full flex-col">
      <TopBar />
      <div className="flex min-h-0 flex-1">
        <Sidebar />
        <main className="flex min-w-0 flex-1 flex-col gap-3 overflow-hidden p-4">
          {error && <p className="rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</p>}
          {!run ? (
            <div className="m-auto max-w-md text-center text-sm text-slate-500">
              Configure a task on the left and start a run. The agent searches the sandbox site in a loop
              (observe → think → act) until it books or runs out of steps; every step is recorded here.
            </div>
          ) : (
            <>
              <RunHeader />
              <nav className="flex gap-1">
                {TABS.map((t) => (
                  <button key={t.id} onClick={() => setTab(t.id)}
                    className={cx("rounded-lg px-3 py-1.5 text-sm font-medium",
                      tab === t.id ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-white")}>
                    {t.label}
                  </button>
                ))}
              </nav>
              <div className="flex min-h-0 flex-1 flex-col">
                {tab === "trajectory" && <TrajectoryView />}
                {tab === "coverage" && <CoverageView />}
                {tab === "fares" && <FaresView />}
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
