import { useEffect, useRef, useState } from "react";
import { api } from "../shared/api";
import { useGym, useSelectedRun } from "../store/gym";
import { Card, PAGE_COLOR, cx } from "./ui";

function StepList() {
  const steps = useGym((s) => s.steps);
  const selectedT = useGym((s) => s.selectedT);
  const selectStep = useGym((s) => s.selectStep);
  const run = useSelectedRun();
  const end = useRef<HTMLLIElement>(null);
  const follow = run?.status === "running" && (selectedT === null || selectedT >= steps.length - 2);

  useEffect(() => {
    if (follow && steps.length) {
      void selectStep(steps.length - 1);
      end.current?.scrollIntoView({ block: "nearest" });
    }
  }, [steps.length, follow, selectStep]);

  return (
    <ol className="h-full overflow-auto scroll-thin">
      {steps.map((s) => (
        <li key={s.t} ref={s.t === steps.length - 1 ? end : undefined}>
          <button onClick={() => void selectStep(s.t)}
            className={cx("flex w-full items-start gap-2 border-b border-slate-100 px-3 py-1.5 text-left hover:bg-slate-50",
              selectedT === s.t && "bg-sky-50")}>
            <span className="mt-1 h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: PAGE_COLOR[s.page] }} />
            <span className="w-8 shrink-0 font-mono text-[11px] text-slate-400">t{s.t}</span>
            <span className="min-w-0 flex-1">
              <span className="block truncate font-mono text-xs text-slate-800">{s.action}</span>
              {s.thought && <span className="block truncate text-[11px] text-slate-500">{s.thought}</span>}
            </span>
            {s.error && <span className="text-[11px] font-medium text-rose-600">error</span>}
          </button>
        </li>
      ))}
      {!steps.length && <li className="p-4 text-sm text-slate-500">Waiting for the first step…</li>}
    </ol>
  );
}

function StepDetail() {
  const run = useSelectedRun();
  const t = useGym((s) => s.selectedT);
  const step = useGym((s) => s.stepDetail);
  const summary = useGym((s) => s.steps.find((x) => x.t === s.selectedT));
  const [showTree, setShowTree] = useState(false);
  const s = step ?? summary;
  if (!run || t === null || !s) return <p className="p-4 text-sm text-slate-500">Select a step.</p>;

  return (
    <div className="h-full space-y-3 overflow-auto p-4 scroll-thin">
      <div className="flex items-center gap-2 text-xs text-slate-500">
        <span className="rounded px-1.5 py-0.5 font-medium text-white" style={{ background: PAGE_COLOR[s.page] }}>{s.page}</span>
        <span className="truncate font-mono">{s.url.replace(/^https?:\/\/[^/]+/, "")}</span>
      </div>
      <img src={api.screenshot(run.id, t)} alt={`Page the agent saw at step ${t}`}
        className="w-full rounded-lg border border-slate-200" />
      {s.thought && (
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Thought</div>
          <p className="text-sm text-slate-800">{s.thought}</p>
        </div>
      )}
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Action</div>
        <code className="block rounded-md bg-slate-900 px-3 py-2 font-mono text-sm text-sky-200">{s.action}</code>
        {s.error && <p className="mt-1 rounded-md bg-rose-50 px-3 py-1.5 font-mono text-xs text-rose-700">{s.error}</p>}
      </div>
      {s.notes && (
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Notes (working memory after this step)</div>
          <pre className="whitespace-pre-wrap rounded-md bg-amber-50 px-3 py-2 font-mono text-xs text-amber-900">{s.notes}</pre>
        </div>
      )}
      {(s.prompt_tokens ?? 0) > 0 && (
        <p className="text-xs text-slate-500">{s.prompt_tokens?.toLocaleString()} prompt tokens · {s.output_tokens} output · {s.latency_ms} ms</p>
      )}
      <div>
        <button onClick={() => setShowTree((v) => !v)} className="text-xs font-medium text-sky-700 hover:underline">
          {showTree ? "Hide" : "Show"} the accessibility tree the agent read
        </button>
        {showTree && (
          <pre className="mt-1 max-h-96 overflow-auto rounded-md bg-slate-50 p-2 font-mono text-[11px] leading-relaxed text-slate-700 scroll-thin">
            {step?.axtree ?? "Loading…"}
          </pre>
        )}
      </div>
    </div>
  );
}

export function TrajectoryView() {
  return (
    <div className="grid min-h-0 flex-1 grid-cols-[minmax(280px,2fr)_3fr] gap-3">
      <Card title="Steps" className="flex min-h-0 flex-col"><div className="min-h-0 flex-1"><StepList /></div></Card>
      <Card title="Step detail" className="flex min-h-0 flex-col"><div className="min-h-0 flex-1"><StepDetail /></div></Card>
    </div>
  );
}
