import { useMemo, useState } from "react";
import { useLab } from "../store/lab";
import { PolicyPicker } from "./EpisodeControls";
import { Badge, Button, Field, Panel, cx, inputCls } from "./ui";

const STARTER = [
  "click-test",
  "click-button",
  "click-link",
  "click-checkboxes",
  "focus-text",
  "enter-text",
  "login-user",
  "choose-list",
  "click-tab",
  "navigate-tree",
].map((t) => `browsergym/miniwob.${t}`);

export function BenchmarkPanel() {
  const tasks = useLab((s) => s.tasks);
  const batch = useLab((s) => s.batch);
  const settings = useLab((s) => s.settings);
  const update = useLab((s) => s.updateSettings);
  const { startBatch, cancelBatch, openReplay } = useLab.getState();

  const benches = Object.keys(tasks);
  const [bench, setBench] = useState("miniwob");
  const [picked, setPicked] = useState<Set<string>>(new Set(STARTER.slice(0, 4)));
  const [filter, setFilter] = useState("");
  const [seedStart, setSeedStart] = useState(0);
  const [nSeeds, setNSeeds] = useState(3);

  const list = (tasks[bench] ?? []).filter((t) => t.includes(filter));
  const toggle = (t: string) =>
    setPicked((p) => {
      const n = new Set(p);
      if (n.has(t)) n.delete(t);
      else n.add(t);
      return n;
    });

  const perTask = useMemo(() => {
    const m = new Map<string, { n: number; ok: number; steps: number }>();
    for (const r of batch?.results ?? []) {
      const e = m.get(r.task) ?? { n: 0, ok: 0, steps: 0 };
      e.n++;
      e.ok += r.success ? 1 : 0;
      e.steps += r.n_steps;
      m.set(r.task, e);
    }
    return [...m.entries()];
  }, [batch]);
  const results = batch?.results ?? [];
  const sr = results.length ? results.filter((r) => r.success).length / results.length : 0;
  const total = picked.size * nSeeds;

  return (
    <div className="grid gap-3 xl:grid-cols-12">
      <Panel className="xl:col-span-5" title="1 · Choose tasks" accent="var(--color-env)" bodyClass="flex flex-col p-3 gap-3">
        <div className="flex gap-2">
          <select className={inputCls} value={bench} onChange={(e) => setBench(e.target.value)}>
            {benches.map((b) => (
              <option key={b}>{b}</option>
            ))}
          </select>
          <input className={cx(inputCls, "flex-1")} placeholder="filter…" value={filter} onChange={(e) => setFilter(e.target.value)} />
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <Button className="!py-1 text-xs" onClick={() => setPicked(new Set(STARTER.filter((t) => tasks[bench]?.includes(t))))}>
            MiniWoB starter set
          </Button>
          <Button className="!py-1 text-xs" onClick={() => setPicked(new Set())}>
            clear
          </Button>
          <span className="self-center text-ink-500">{picked.size} selected</span>
        </div>
        <div className="h-[380px] overflow-auto rounded border border-ink-700 bg-ink-950 p-1 scroll-thin">
          {list.map((t) => (
            <label key={t} className="flex cursor-pointer items-center gap-2 rounded px-2 py-0.5 font-mono text-[11.5px] hover:bg-ink-850">
              <input type="checkbox" checked={picked.has(t)} onChange={() => toggle(t)} className="accent-amber-500" />
              {t.replace("browsergym/", "")}
            </label>
          ))}
        </div>
        {bench !== "miniwob" && (
          <p className="text-xs text-act">
            Non-MiniWoB benchmarks need their own servers and environment variables (see the BrowserGym README for {bench}).
          </p>
        )}
      </Panel>

      <div className="flex flex-col gap-3 xl:col-span-7">
        <Panel title="2 · Configure the rollout" accent="var(--color-policy)" bodyClass="p-3">
          <div className="flex flex-wrap items-end gap-4">
            <PolicyPicker allowHuman={false} />
            <Field label="Seeds">
              <div className="flex items-center gap-1 text-xs text-ink-500">
                <input className={cx(inputCls, "w-16")} type="number" min={0} value={seedStart} onChange={(e) => setSeedStart(Number(e.target.value))} />
                <span>×</span>
                <input className={cx(inputCls, "w-14")} type="number" min={1} max={50} value={nSeeds} onChange={(e) => setNSeeds(Math.max(1, Number(e.target.value)))} />
              </div>
            </Field>
            <Field label="Max steps">
              <input className={cx(inputCls, "w-16")} type="number" min={1} max={50} value={settings.maxSteps} onChange={(e) => update({ maxSteps: Number(e.target.value) })} />
            </Field>
            {batch?.running ? (
              <Button variant="danger" onClick={cancelBatch}>
                Cancel
              </Button>
            ) : (
              <Button
                variant="policy"
                disabled={!picked.size}
                onClick={() => startBatch([...picked], Array.from({ length: nSeeds }, (_, i) => seedStart + i))}
              >
                Run {total} episode{total === 1 ? "" : "s"}
              </Button>
            )}
          </div>
          <p className="mt-3 text-xs leading-relaxed text-ink-500">
            Each (task, seed) pair is one episode run without supervision. Every episode is saved, so a benchmark run doubles as a
            data-collection run: this is how an LLM policy generates trajectories at scale. Raise the temperature to get more diverse
            rollouts of the same task.
          </p>
        </Panel>

        <Panel
          title="3 · Results"
          accent="var(--color-reward)"
          bodyClass="p-3"
          right={batch && <span className="font-mono text-[11px] text-ink-300">{batch.done}/{batch.total}{batch.running ? " running…" : ""}</span>}
        >
          {!batch ? (
            <p className="text-sm text-ink-500">No run yet.</p>
          ) : (
            <div className="space-y-4">
              <div className="h-1.5 overflow-hidden rounded bg-ink-800">
                <div className="h-full bg-policy transition-all" style={{ width: `${(100 * batch.done) / Math.max(1, batch.total)}%` }} />
              </div>
              <div className="flex gap-6">
                <Stat label="Success rate" value={`${(sr * 100).toFixed(0)}%`} color="var(--color-reward)" />
                <Stat label="Episodes" value={String(results.length)} />
                <Stat label="Mean steps" value={results.length ? (results.reduce((a, r) => a + r.n_steps, 0) / results.length).toFixed(1) : "–"} />
              </div>
              <div className="space-y-1.5">
                {perTask.map(([task, e]) => (
                  <div key={task} className="flex items-center gap-3 font-mono text-[11px]">
                    <span className="w-48 truncate text-ink-300">{task.replace("browsergym/miniwob.", "")}</span>
                    <div className="h-3 flex-1 overflow-hidden rounded-sm bg-ink-800">
                      <div className="h-full bg-reward/80" style={{ width: `${(100 * e.ok) / e.n}%` }} />
                    </div>
                    <span className="w-16 text-right text-ink-300">
                      {e.ok}/{e.n}
                    </span>
                  </div>
                ))}
              </div>
              <div className="max-h-52 overflow-auto rounded border border-ink-700 scroll-thin">
                <table className="w-full font-mono text-[11px]">
                  <tbody>
                    {results.map((r) => (
                      <tr key={r.id} className="cursor-pointer border-b border-ink-800 hover:bg-ink-850" onClick={() => void openReplay(r.id)}>
                        <td className="px-2 py-1 text-ink-300">{r.task.replace("browsergym/", "")}</td>
                        <td className="px-2 py-1 text-ink-500">seed {r.seed}</td>
                        <td className="px-2 py-1 text-ink-500">{r.n_steps} steps</td>
                        <td className="px-2 py-1 text-right">
                          <Badge color={r.success ? "var(--color-reward)" : "var(--color-act)"}>{r.success ? "pass" : "fail"}</Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[11px] text-ink-500">Click a row to replay that episode step by step.</p>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-500">{label}</div>
      <div className="font-mono text-2xl font-semibold" style={{ color }}>
        {value}
      </div>
    </div>
  );
}
