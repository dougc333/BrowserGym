import { useMemo, useState } from "react";
import { useLab, useViewedStep } from "../store/lab";
import { actionBid } from "../lib/bc";
import { Panel, cx } from "./ui";

type View = "axtree" | "screenshot" | "prompt" | "response";
const LINE_RE = /^(\s*)\[([^\]]+)\]\s+(\S+)(.*)$/;
const TEXT_ROLES = new Set(["textbox", "searchbox", "combobox", "spinbutton"]);

export function ObservationPanel({ className }: { className?: string }) {
  const [view, setView] = useState<View>("axtree");
  const [filter, setFilter] = useState("");
  const viewed = useViewedStep();
  const liveObs = useLab((s) => s.obs);
  const liveProposal = useLab((s) => s.proposal);
  const stream = useLab((s) => s.stream);
  const draft = useLab((s) => s.draft);
  const setDraft = useLab((s) => s.setDraft);
  const phase = useLab((s) => s.phase);
  const config = useLab((s) => s.config);

  const obs = viewed ? viewed.obs : liveObs;
  const proposal = viewed ? viewed.proposal : liveProposal;
  const prompt = viewed ? (viewed.prompt ?? viewed.proposal?.prompt) : liveProposal?.prompt;
  const highlightBid = actionBid(viewed ? viewed.action : draft || liveProposal?.action);
  const canPick = !viewed && (phase === "observing" || phase === "awaiting_approval");

  const lines = useMemo(() => (obs ? obs.axtree.split("\n") : []), [obs]);

  const pick = (bid: string, role: string) =>
    setDraft(TEXT_ROLES.has(role) ? `fill("${bid}", "")` : `click("${bid}")`);

  return (
    <Panel
      className={cx("min-h-[420px]", className)}
      title={viewed ? `Observation o${sub(viewed.t)} (recorded)` : "Observation oₜ (live)"}
      accent="var(--color-obs)"
      bodyClass="flex flex-col"
      right={
        <div className="flex gap-1">
          {(["axtree", "screenshot", "prompt", "response"] as View[]).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={cx(
                "rounded px-2 py-0.5 font-mono text-[11px]",
                view === v ? "bg-ink-700 text-ink-100" : "text-ink-500 hover:text-ink-100",
              )}
            >
              {v}
            </button>
          ))}
        </div>
      }
    >
      {!obs ? (
        <Empty />
      ) : (
        <>
          <div className="border-b border-ink-700 px-3 py-2.5">
            <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-500">Goal</div>
            <div className="mt-0.5 text-[15px] font-medium leading-snug">{obs.goal}</div>
            <div className="mt-1 truncate font-mono text-[10px] text-ink-500">{obs.url}</div>
            {obs.last_action_error && (
              <div className="mt-1.5 rounded border border-act/40 bg-act/10 px-2 py-1 font-mono text-[11px] text-act">
                last_action_error: {obs.last_action_error}
              </div>
            )}
          </div>

          {view === "axtree" && (
            <div className="flex min-h-0 flex-1 flex-col">
              <div className="flex items-center gap-2 px-3 py-1.5">
                <input
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  placeholder="filter nodes…"
                  className="w-40 rounded border border-ink-700 bg-ink-950 px-2 py-0.5 font-mono text-[11px] outline-none focus:border-obs"
                />
                <span className="text-[11px] text-ink-500">
                  {canPick ? "Click a [bid] to write an action for it" : "This is the exact text Claude reads"}
                </span>
              </div>
              <pre className="min-h-0 flex-1 overflow-auto px-3 pb-3 font-mono text-[11.5px] leading-[1.55] scroll-thin">
                {lines.map((line, i) => {
                  if (filter && !line.toLowerCase().includes(filter.toLowerCase())) return null;
                  const m = line.match(LINE_RE);
                  if (!m) return <div key={i} className="whitespace-pre text-ink-500">{line}</div>;
                  const [, indent, bid, role, rest] = m;
                  const hot = bid === highlightBid;
                  return (
                    <div key={i} className={cx("whitespace-pre rounded-sm", hot && "bg-act/15 outline outline-1 outline-act/60")}>
                      {indent}
                      <button
                        disabled={!canPick}
                        onClick={() => pick(bid, role)}
                        className={cx(
                          "rounded px-0.5 font-bold",
                          hot ? "text-act" : "text-obs",
                          canPick && "hover:bg-obs/20 hover:underline",
                        )}
                      >
                        [{bid}]
                      </button>{" "}
                      <span className="text-env">{role}</span>
                      <span className="text-ink-300">{rest}</span>
                    </div>
                  );
                })}
              </pre>
            </div>
          )}

          {view === "screenshot" && (
            <div className="min-h-0 flex-1 overflow-auto p-3 scroll-thin">
              {obs.screenshot ? (
                <img src={obs.screenshot} alt="Page screenshot" className="max-w-full rounded border border-ink-700" />
              ) : (
                <p className="text-sm text-ink-500">No screenshot captured.</p>
              )}
              <p className="mt-2 text-xs text-ink-500">
                The screenshot is only sent to Claude when “Send screenshot to Claude” is on. By default the policy acts from text alone.
              </p>
            </div>
          )}

          {view === "prompt" && (
            <pre className="min-h-0 flex-1 overflow-auto whitespace-pre-wrap p-3 font-mono text-[11px] leading-relaxed text-ink-300 scroll-thin">
              {prompt ? (
                <>
                  <span className="text-policy"># SYSTEM{"\n"}</span>
                  {prompt.system}
                  {"\n\n"}
                  <span className="text-policy"># USER{"\n"}</span>
                  {prompt.user}
                </>
              ) : (
                <>
                  <span className="text-ink-500">
                    The full prompt appears once the policy is queried. System prompt:{"\n\n"}
                  </span>
                  {config?.system_prompt}
                </>
              )}
            </pre>
          )}

          {view === "response" && (
            <pre className="min-h-0 flex-1 overflow-auto whitespace-pre-wrap p-3 font-mono text-[12px] leading-relaxed scroll-thin">
              {viewed ? viewed.proposal?.raw || "(human step: no model response)" : stream || proposal?.raw || "(no response yet)"}
            </pre>
          )}
        </>
      )}
    </Panel>
  );
}

const sub = (n: number) => String(n).replace(/\d/g, (d) => "₀₁₂₃₄₅₆₇₈₉"[Number(d)]);

function Empty() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 p-8 text-center">
      <div className="font-mono text-3xl text-ink-700">[ ]</div>
      <p className="max-w-xs text-sm text-ink-500">
        Pick a task and press <span className="font-mono text-ink-300">env.reset()</span>. BrowserGym launches Chromium,
        loads the task, and returns the first observation as an accessibility tree.
      </p>
    </div>
  );
}
