import { isBusy, useLab, useViewedStep } from "../store/lab";
import { Badge, Button, Panel, cx, sourceColor } from "./ui";

const QUICK = ['noop()', 'scroll(0, 300)', 'go_back()', 'send_msg_to_user("")'];

export function PolicyPanel() {
  const viewed = useViewedStep();
  return (
    <Panel
      className="min-h-[260px] flex-1"
      title={viewed ? `Decision at t=${viewed.t}` : "Policy output π(aₜ | oₜ)"}
      accent="var(--color-policy)"
      bodyClass="flex flex-col"
    >
      {viewed ? <RecordedDecision /> : <LiveDecision />}
    </Panel>
  );
}

function LiveDecision() {
  const phase = useLab((s) => s.phase);
  const stream = useLab((s) => s.stream);
  const proposal = useLab((s) => s.proposal);
  const draft = useLab((s) => s.draft);
  const setDraft = useLab((s) => s.setDraft);
  const settings = useLab((s) => s.settings);
  const autopilot = useLab((s) => s.autopilot);
  const { think, execute, run, pause } = useLab.getState();

  const hasEpisode = phase !== "idle" && phase !== "done";
  const busy = isBusy(phase);
  const edited = !!proposal && draft.trim() !== proposal.action.trim();
  const canAct = hasEpisode && !busy && draft.trim().length > 0;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 overflow-auto px-3 py-2.5 scroll-thin">
        {phase === "thinking" ? (
          <pre className="caret whitespace-pre-wrap font-mono text-[12px] leading-relaxed text-ink-300">{stream}</pre>
        ) : proposal ? (
          <div className="space-y-2.5">
            <div className="flex items-center gap-2">
              <Badge color={sourceColor(proposal.policy)}>{proposal.policy}</Badge>
              {proposal.model && <span className="font-mono text-[10px] text-ink-500">{proposal.model}</span>}
              {proposal.latency_ms > 0 && (
                <span className="font-mono text-[10px] text-ink-500">
                  {proposal.latency_ms} ms · {proposal.usage.input_tokens}→{proposal.usage.output_tokens} tok
                </span>
              )}
            </div>
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-500">Thought</div>
              <p className="mt-0.5 text-[13px] leading-relaxed text-ink-100">{proposal.thought || "—"}</p>
            </div>
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-500">Proposed action</div>
              <code className="mt-0.5 block font-mono text-[13px] text-policy">{proposal.action}</code>
            </div>
          </div>
        ) : (
          <p className="text-sm leading-relaxed text-ink-500">
            {phase === "idle"
              ? "No episode yet. Reset an environment to get the first observation."
              : phase === "done"
                ? "Episode over. Reset for a new one, or open the Trajectories tab to turn it into training data."
                : settings.policy === "human"
                  ? "You are the policy. Click a [bid] in the AXTree or type an action below. Human demonstrations are recorded just like Claude's."
                  : "Press Propose to query the policy once, or Autopilot to let it run the loop."}
          </p>
        )}
      </div>

      <div className="border-t border-ink-700 p-3">
        <div className="mb-1.5 flex items-center justify-between">
          <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-500">
            Action to execute
          </span>
          {edited && <Badge color="var(--color-act)">human correction</Badge>}
        </div>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && canAct && execute()}
          placeholder={hasEpisode ? 'click("13")' : "reset first"}
          disabled={!hasEpisode}
          className={cx(
            "w-full rounded-md border bg-ink-950 px-2.5 py-2 font-mono text-[13px] outline-none",
            edited ? "border-act text-act" : "border-ink-700 text-policy focus:border-policy",
          )}
        />
        <div className="mt-1.5 flex flex-wrap gap-1">
          {QUICK.map((q) => (
            <button
              key={q}
              disabled={!hasEpisode}
              onClick={() => setDraft(q)}
              className="rounded border border-ink-700 px-1.5 py-0.5 font-mono text-[10px] text-ink-500 hover:text-ink-100 disabled:opacity-30"
            >
              {q}
            </button>
          ))}
        </div>
        <div className="mt-2.5 flex flex-wrap gap-2">
          <Button
            variant="policy"
            onClick={think}
            disabled={!hasEpisode || busy || settings.policy === "human"}
            title="Query the policy once and show its proposal (no env step yet)"
          >
            Propose
          </Button>
          <Button variant="primary" onClick={execute} disabled={!canAct} title="env.step(action)">
            {edited ? "Execute correction" : proposal ? "Approve & step" : "Execute"}
          </Button>
          {autopilot ? (
            <Button variant="danger" onClick={pause}>
              Pause
            </Button>
          ) : (
            <Button
              onClick={run}
              disabled={!hasEpisode || busy || settings.policy === "human"}
              title={settings.supervised ? "Run the loop, stopping for your approval each step" : "Run the loop until the episode ends"}
            >
              Autopilot {settings.supervised ? "(supervised)" : "▶"}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

function RecordedDecision() {
  const step = useViewedStep()!;
  const p = step.proposal;
  const overridden = !!p && p.action !== step.action;
  return (
    <div className="space-y-3 overflow-auto px-3 py-2.5 scroll-thin">
      <div className="flex flex-wrap items-center gap-2">
        <Badge color={sourceColor(step.action_source)}>decided by {step.action_source}</Badge>
        <Badge color={step.reward > 0 ? "var(--color-reward)" : undefined}>r = {step.reward}</Badge>
        {step.terminated && <Badge color="var(--color-reward)">terminated</Badge>}
        {step.truncated && <Badge>truncated</Badge>}
      </div>
      {step.thought && (
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-500">Thought</div>
          <p className="mt-0.5 text-[13px] leading-relaxed">{step.thought}</p>
        </div>
      )}
      {overridden && (
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-500">Policy proposed (rejected)</div>
          <code className="mt-0.5 block font-mono text-[13px] text-ink-500 line-through">{p!.action}</code>
        </div>
      )}
      <div>
        <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-500">Executed action (BC label)</div>
        <code className="mt-0.5 block font-mono text-[13px] text-policy">{step.action}</code>
      </div>
      {step.action_error && (
        <div className="rounded border border-act/40 bg-act/10 px-2 py-1 font-mono text-[11px] text-act">{step.action_error}</div>
      )}
      <p className="text-xs text-ink-500">
        In the dataset this step becomes one example: the observation on the left is the input, the executed action is the target.
        {overridden && " Because a human overrode the policy, it also yields a preference pair (chosen vs rejected)."}
      </p>
    </div>
  );
}
