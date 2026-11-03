import { useGym, useSelectedRun } from "../store/gym";
import type { PageType } from "../shared/schemas";
import { Button, Card, PAGE_COLOR, Pill, Stat, cx, money } from "./ui";

/** One tick per step, colored by page type: the shape of a long search at a glance. */
function TrajectoryStrip() {
  const steps = useGym((s) => s.steps);
  const selectedT = useGym((s) => s.selectedT);
  const selectStep = useGym((s) => s.selectStep);
  const setTab = useGym((s) => s.setTab);
  if (!steps.length) return null;
  return (
    <div>
      <div className="flex h-7 w-full overflow-hidden rounded-md border border-slate-200">
        {steps.map((s) => (
          <button key={s.t} title={`t${s.t} · ${s.page} · ${s.action}`}
            onClick={() => { setTab("trajectory"); void selectStep(s.t); }}
            className={cx("h-full min-w-[2px] flex-1 border-r border-white/60", s.error && "opacity-50",
              selectedT === s.t && "ring-2 ring-inset ring-slate-900")}
            style={{ background: PAGE_COLOR[s.page] }} />
        ))}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-3 text-[11px] text-slate-500">
        {(["search", "results", "details", "review", "confirmation"] as PageType[]).map((p) => (
          <span key={p} className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-sm" style={{ background: PAGE_COLOR[p] }} />
            {p} {steps.filter((s) => s.page === p).length}
          </span>
        ))}
      </div>
    </div>
  );
}

export function RunHeader() {
  const run = useSelectedRun();
  const cov = useGym((s) => s.coverage);
  const stop = useGym((s) => s.stop);
  if (!run) return null;
  const o = run.outcome;
  const live = run.status === "running" || run.status === "queued";
  const elapsed = run.started_at ? ((run.ended_at ?? Date.now() / 1000) - run.started_at).toFixed(0) : "–";

  return (
    <div className="space-y-3">
      <Card>
        <div className="space-y-3 p-4">
          <div className="flex items-start gap-3">
            <p className="flex-1 text-[15px] leading-snug text-slate-800">{cov?.task.goal ?? "Generating the episode…"}</p>
            {live && <Button variant="danger" onClick={() => void stop()}>Stop</Button>}
          </div>
          <div className="flex flex-wrap items-end gap-x-8 gap-y-3">
            <Stat label="Status" value={
              run.status === "done" && o ? (
                <span className={o.success ? "text-emerald-600" : "text-rose-600"}>{o.success ? "Optimal" : o.booked ? "Suboptimal" : "No booking"}</span>
              ) : <span className={run.status === "error" ? "text-rose-600" : "text-amber-600"}>{run.status}</span>
            } sub={o?.reason} />
            <Stat label="Channel" value={run.policy.startsWith("mcp") ? "MCP tools" : "Browser"}
              sub={run.model ?? run.policy.replace("mcp-", "")} />
            <Stat label="Steps" value={`${run.n_steps} / ${run.max_steps}`} sub={`${elapsed}s`} />
            <Stat label="Booked" value={money(o?.true_total_for_task ?? o?.booked_total)} sub={o?.booked ? `flight #${o.booked_flight_id}` : "–"} />
            <Stat label="Optimum" value={money(o?.optimal_total ?? cov?.optimum.total)}
              sub={cov ? `${cov.optimum.origin} · ${cov.optimum.date} · page ${cov.optimum.page}` : undefined} />
            {o?.regret != null && <Stat label="Overpaid" value={money(o.regret)} />}
            {!run.policy.endsWith("scripted") && (
              <Stat label="Tokens" value={(run.input_tokens + run.output_tokens).toLocaleString()}
                sub={`${run.input_tokens.toLocaleString()} in · ${run.output_tokens.toLocaleString()} out`} />
            )}
          </div>
          {run.error && <p className="rounded-md bg-rose-50 px-3 py-2 font-mono text-xs text-rose-700">{run.error}</p>}
          <TrajectoryStrip />
        </div>
      </Card>
      {o?.booked && (
        <div className="flex flex-wrap gap-1.5">
          {o.constraints.map((c) => (
            <Pill key={c.name} tone={c.ok ? "green" : "red"}>
              {c.ok ? "✓" : "✗"} {c.name}: {String(c.actual)}{!c.ok && ` (needs ${c.expected})`}
            </Pill>
          ))}
        </div>
      )}
    </div>
  );
}
