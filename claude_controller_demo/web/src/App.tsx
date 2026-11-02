import { useLab } from "./store/lab";
import { TopBar } from "./components/TopBar";
import { EpisodeControls } from "./components/EpisodeControls";
import { LoopDiagram } from "./components/LoopDiagram";
import { ObservationPanel } from "./components/ObservationPanel";
import { PolicyPanel } from "./components/PolicyPanel";
import { Timeline } from "./components/Timeline";
import { BenchmarkPanel } from "./components/BenchmarkPanel";
import { DatasetPanel } from "./components/DatasetPanel";
import { LearnPanel } from "./components/LearnPanel";
import { LogStrip } from "./components/LogStrip";

export default function App() {
  const tab = useLab((s) => s.tab);
  return (
    <div className="flex h-full flex-col">
      <TopBar />
      <main className="min-h-0 flex-1 overflow-auto p-3 scroll-thin">
        {tab === "live" && (
          <div className="flex h-full min-h-[720px] flex-col gap-3">
            <EpisodeControls />
            <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 xl:grid-cols-12">
              <ObservationPanel className="xl:col-span-5" />
              <div className="flex min-h-0 flex-col gap-3 xl:col-span-4">
                <LoopDiagram />
                <PolicyPanel />
              </div>
              <Timeline className="xl:col-span-3" />
            </div>
          </div>
        )}
        {tab === "bench" && <BenchmarkPanel />}
        {tab === "data" && <DatasetPanel />}
        {tab === "learn" && <LearnPanel />}
      </main>
      <LogStrip />
    </div>
  );
}
