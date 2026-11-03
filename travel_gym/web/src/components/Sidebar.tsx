import { useGym } from "../store/gym";
import type { Difficulty } from "../shared/schemas";
import { Button, Pill, cx } from "./ui";

const NUM_KNOBS: { key: keyof Difficulty; label: string; hint: string; min: number; max: number }[] = [
  { key: "n_dates", label: "Dates", hint: "Departure days the user allows: more searches", min: 1, max: 14 },
  { key: "n_origins", label: "Airports", hint: "Nearby origin airports allowed: more searches", min: 1, max: 3 },
  { key: "pages", label: "Pages", hint: "Result pages per search: more paging", min: 1, max: 8 },
  { key: "page_size", label: "Per page", hint: "Results shown per page", min: 3, max: 20 },
  { key: "passengers", label: "Max travelers", hint: "Total price scales with travelers", min: 1, max: 4 },
  { key: "decoys", label: "Decoys", hint: "Cheaper flights that each break one requirement", min: 0, max: 20 },
];
const BOOL_KNOBS: { key: keyof Difficulty; label: string; hint: string }[] = [
  { key: "hidden_fees", label: "Hidden fees", hint: "Bag fees and taxes only on details / review pages" },
  { key: "sort_available", label: "Sort by price", hint: "Results can be sorted by listed fare" },
  { key: "bags", label: "Checked bag", hint: "Task requires one checked bag per traveler" },
  { key: "require_nonstop", label: "Nonstop", hint: "Task requires nonstop" },
  { key: "depart_after", label: "Depart after", hint: "Task requires a minimum departure time" },
  { key: "require_refundable", label: "Refundable", hint: "Task requires a refundable fare" },
];

function NewRunForm() {
  const form = useGym((s) => s.form);
  const config = useGym((s) => s.config);
  const error = useGym((s) => s.formError);
  const { setForm, setDifficulty, applyPreset, start } = useGym.getState();
  const d = form.difficulty;
  const searches = d.n_dates * d.n_origins;

  return (
    <div className="space-y-3 p-4">
      <div className="flex flex-wrap gap-1.5">
        {Object.keys(config?.presets ?? {}).map((p) => (
          <button key={p} onClick={() => applyPreset(p)}
            className="rounded-md border border-slate-300 px-2 py-0.5 text-xs capitalize text-slate-600 hover:border-sky-500 hover:text-sky-700">
            {p}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-2">
        {NUM_KNOBS.map((k) => (
          <label key={k.key} title={k.hint} className="text-[11px] text-slate-500">
            {k.label}
            <input type="number" min={k.min} max={k.max} value={d[k.key] as number}
              onChange={(e) => setDifficulty({ [k.key]: Number(e.target.value) } as Partial<Difficulty>)}
              className="mt-0.5 w-full rounded-md border border-slate-300 px-2 py-1 text-sm text-slate-900" />
          </label>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-x-2 gap-y-1">
        {BOOL_KNOBS.map((k) => (
          <label key={k.key} title={k.hint} className="flex items-center gap-1.5 text-xs text-slate-600">
            <input type="checkbox" checked={d[k.key] as boolean} className="accent-sky-600"
              onChange={(e) => setDifficulty({ [k.key]: e.target.checked } as Partial<Difficulty>)} />
            {k.label}
          </label>
        ))}
      </div>
      <p className="text-xs text-slate-500">
        {searches} search{searches > 1 ? "es" : ""} × up to {d.pages} page{d.pages > 1 ? "s" : ""} ≈{" "}
        {searches * d.pages * d.page_size} flights to consider.
      </p>
      <div className="grid grid-cols-3 gap-2">
        <label className="text-[11px] text-slate-500">
          Seed
          <input type="number" min={0} value={form.seed} onChange={(e) => setForm({ seed: Number(e.target.value) })}
            className="mt-0.5 w-full rounded-md border border-slate-300 px-2 py-1 text-sm text-slate-900" />
        </label>
        <label className="text-[11px] text-slate-500">
          Max steps
          <input type="number" min={1} max={1000} value={form.max_steps}
            onChange={(e) => setForm({ max_steps: Number(e.target.value) })}
            className="mt-0.5 w-full rounded-md border border-slate-300 px-2 py-1 text-sm text-slate-900" />
        </label>
        <label className="text-[11px] text-slate-500">
          Policy
          <select value={form.policy} onChange={(e) => setForm({ policy: e.target.value as typeof form.policy })}
            className="mt-0.5 w-full rounded-md border border-slate-300 px-1 py-1 text-sm text-slate-900">
            <option value="scripted">Scripted</option>
            <option value="claude" disabled={!config?.has_api_key}>Claude</option>
            <option value="openai">OpenAI-compat</option>
          </select>
        </label>
      </div>
      {form.policy === "claude" && (
        <select value={form.model ?? config?.default_model} onChange={(e) => setForm({ model: e.target.value })}
          className="w-full rounded-md border border-slate-300 px-2 py-1 text-sm">
          {config?.models.map((m) => <option key={m}>{m}</option>)}
        </select>
      )}
      {form.policy === "openai" && (
        <input placeholder="qwen2.5:7b-instruct@http://localhost:11434/v1" value={form.model ?? ""}
          onChange={(e) => setForm({ model: e.target.value })}
          className="w-full rounded-md border border-slate-300 px-2 py-1 font-mono text-xs" />
      )}
      <label className="flex items-center gap-1.5 text-xs text-slate-600">
        <input type="checkbox" checked={!form.headless} className="accent-sky-600"
          onChange={(e) => setForm({ headless: !e.target.checked })} />
        Show the Chromium window
      </label>
      {error && <p className="rounded-md bg-rose-50 px-2 py-1 text-xs text-rose-700">{error}</p>}
      <Button variant="primary" className="w-full" onClick={() => void start()}>Start run</Button>
    </div>
  );
}

function RunsList() {
  const runs = useGym((s) => s.runs);
  const selected = useGym((s) => s.selectedId);
  const select = useGym((s) => s.select);
  return (
    <ul className="min-h-0 flex-1 overflow-auto scroll-thin">
      {runs.map((r) => {
        const d = r.difficulty;
        return (
          <li key={r.id}>
            <button onClick={() => void select(r.id)}
              className={cx("w-full border-b border-slate-100 px-4 py-2 text-left hover:bg-slate-50",
                selected === r.id && "bg-sky-50")}>
              <div className="flex items-center gap-2">
                <span className="font-mono text-[11px] text-slate-500">{r.id.slice(9)}</span>
                <Pill tone={r.policy === "claude" ? "violet" : r.policy === "openai" ? "sky" : "slate"}>{r.model ?? r.policy}</Pill>
                <span className="ml-auto">
                  {r.status === "done" ? (
                    <Pill tone={r.success ? "green" : "red"}>{r.success ? "optimal" : r.outcome?.booked ? "suboptimal" : "no booking"}</Pill>
                  ) : (
                    <Pill tone={r.status === "error" ? "red" : "amber"}>{r.status}</Pill>
                  )}
                </span>
              </div>
              <div className="mt-0.5 text-xs text-slate-500">
                {d.n_dates}d × {d.n_origins}ap × {d.pages}p{d.hidden_fees ? " · hidden fees" : ""} · seed {r.seed} ·{" "}
                {r.n_steps} steps
              </div>
            </button>
          </li>
        );
      })}
      {!runs.length && <li className="px-4 py-3 text-sm text-slate-500">No runs yet.</li>}
    </ul>
  );
}

export function Sidebar() {
  return (
    <aside className="flex w-[340px] shrink-0 flex-col border-r border-slate-200 bg-white">
      <div className="border-b border-slate-200 px-4 pt-3 text-xs font-semibold uppercase tracking-wider text-slate-500">New run</div>
      <NewRunForm />
      <div className="border-y border-slate-200 bg-slate-50 px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500">Runs</div>
      <RunsList />
    </aside>
  );
}
