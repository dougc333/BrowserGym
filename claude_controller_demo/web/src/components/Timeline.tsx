import { useLab, useViewedSteps } from "../store/lab";
import { returnsToGo } from "../lib/bc";
import { Badge, Button, Panel, cx, sourceColor } from "./ui";

export function Timeline({ className }: { className?: string }) {
  const steps = useViewedSteps();
  const replay = useLab((s) => s.replay);
  const live = useLab((s) => s.episode);
  const phase = useLab((s) => s.phase);
  const selected = useLab((s) => s.selectedStep);
  const gamma = useLab((s) => s.settings.gamma);
  const { selectStep, closeReplay } = useLab.getState();

  const episode = replay ?? live;
  const rtg = returnsToGo(steps, gamma);
  const ended = replay ? true : phase === "done";

  return (
    <Panel
      className={cx("min-h-[420px]", className)}
      title="Trajectory τ"
      accent="var(--color-reward)"
      bodyClass="flex flex-col"
      right={<span className="font-mono text-[10px] text-ink-500">γ = {gamma}</span>}
    >
      {replay && (
        <div className="flex items-center gap-2 border-b border-ink-700 bg-obs/10 px-3 py-1.5 text-xs">
          <span className="text-obs">Replaying saved episode</span>
          <Button className="ml-auto !px-2 !py-0.5 text-xs" onClick={closeReplay}>
            back to live
          </Button>
        </div>
      )}
      {episode ? (
        <div className="border-b border-ink-700 px-3 py-2">
          <div className="truncate font-mono text-[11px] text-ink-300">{episode.task.replace("browsergym/", "")}</div>
          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            <Badge>seed {episode.seed}</Badge>
            <Badge>{steps.length} steps</Badge>
            <Badge color="var(--color-reward)">G₀ = {(rtg[0] ?? 0).toFixed(3)}</Badge>
            {ended && (
              <Badge color={episode.success ? "var(--color-reward)" : "var(--color-act)"}>
                {episode.success ? "success" : "failure"}
              </Badge>
            )}
          </div>
        </div>
      ) : null}

      <ol className="min-h-0 flex-1 overflow-auto p-2 scroll-thin">
        {steps.map((s, i) => (
          <li key={i}>
            <button
              onClick={() => selectStep(selected === i ? (replay ? 0 : null) : i)}
              className={cx(
                "group mb-1 w-full rounded-md border px-2 py-1.5 text-left transition",
                selected === i ? "border-obs bg-obs/10" : "border-transparent hover:border-ink-700 hover:bg-ink-850",
              )}
            >
              <div className="flex items-center gap-2">
                <span className="w-6 font-mono text-[10px] text-ink-500">t{s.t}</span>
                <span
                  className="h-1.5 w-1.5 shrink-0 rounded-full"
                  style={{ background: sourceColor(s.action_source) }}
                  title={`decided by ${s.action_source}`}
                />
                <code className="truncate font-mono text-[11.5px] text-ink-100">{s.action}</code>
              </div>
              <div className="mt-0.5 flex items-center gap-2 pl-8 font-mono text-[10px]">
                <span className={s.reward > 0 ? "text-reward" : "text-ink-500"}>r={s.reward}</span>
                <span className="text-ink-500">G={rtg[i].toFixed(3)}</span>
                {s.proposal && s.proposal.action !== s.action && <span className="text-act">corrected</span>}
                {s.action_error && <span className="text-act">error</span>}
                {(s.terminated || s.truncated) && <span className="text-ink-300">{s.terminated ? "terminal" : "truncated"}</span>}
              </div>
            </button>
          </li>
        ))}
        {!replay && live && !ended && (
          <li>
            <button
              onClick={() => selectStep(null)}
              className={cx(
                "w-full rounded-md border border-dashed px-2 py-1.5 text-left font-mono text-[11px]",
                selected === null ? "border-policy text-policy" : "border-ink-700 text-ink-500",
              )}
            >
              t{steps.length} · now ({phase.replace("_", " ")})
            </button>
          </li>
        )}
        {!episode && <p className="p-2 text-sm text-ink-500">Transitions appear here as the policy acts.</p>}
      </ol>
      <p className="border-t border-ink-700 px-3 py-2 text-[11px] leading-relaxed text-ink-500">
        Click a step to inspect exactly what the policy saw and did. G is the discounted return-to-go; MiniWoB rewards are sparse, so
        credit only appears at the end.
      </p>
    </Panel>
  );
}
