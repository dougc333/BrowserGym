import type { ReactNode } from "react";
import { useLab, type Tab } from "../store/lab";
import { Panel } from "./ui";

const MAPPING: [string, string, string][] = [
  ["State sₜ", "The full browser: DOM, JS state, server state", "Hidden. The policy never sees it directly, so this is a POMDP."],
  ["Observation oₜ", "Goal + flattened AXTree (+ optional screenshot)", "BrowserGym's obs['axtree_object'] → flatten_axtree_to_str()"],
  ["Action aₜ", 'A string like click("13") or fill("21", "hello")', "HighLevelActionSet: bid, nav, tab, chat, infeas subsets"],
  ["Policy π(a|o)", "Claude, prompted with oₜ and the action space", "Sampling temperature ≈ exploration. No weights change."],
  ["Transition", "env.step(a) runs the action in Chromium via Playwright", "Returns oₜ₊₁, rₜ, terminated, truncated"],
  ["Reward rₜ", "Task validator: 1.0 on success, 0 otherwise", "Sparse and delayed: credit only at the end"],
  ["Episode / τ", "reset → steps until terminated or the step budget", "Saved to runs/<id>.json, one per episode"],
  ["Return Gₜ", "Σₖ γᵏ rₜ₊ₖ, shown in the timeline", "Used to filter data or condition a Decision Transformer"],
];

export function LearnPanel() {
  const setTab = useLab((s) => s.setTab);
  return (
    <div className="mx-auto grid max-w-6xl gap-3 lg:grid-cols-2">
      <Panel className="lg:col-span-2" title="The map: RL vocabulary ↔ this demo" accent="var(--color-policy)" bodyClass="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-ink-700 text-left text-[10px] uppercase tracking-[0.12em] text-ink-500">
              <th className="px-3 py-2">RL concept</th>
              <th className="px-3 py-2">In BrowserGym + Claude</th>
              <th className="px-3 py-2">Where to look</th>
            </tr>
          </thead>
          <tbody>
            {MAPPING.map(([a, b, c]) => (
              <tr key={a} className="border-b border-ink-800">
                <td className="px-3 py-2 font-mono text-policy">{a}</td>
                <td className="px-3 py-2">{b}</td>
                <td className="px-3 py-2 text-ink-500">{c}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      <Card title="Why the accessibility tree?" color="var(--color-obs)">
        <p>
          A screenshot is pixels; the policy would have to output coordinates. The AXTree is the page as assistive technology sees it:
          roles, names and states, each interactive node tagged with a stable <code>[bid]</code>. That gives the LLM a compact, textual
          state and a discrete, nameable action target, which is exactly what a language model is good at.
        </p>
        <p>
          Try the <b>Human</b> policy on the Live tab and click a few bids yourself: you are now acting from the same observation Claude
          gets.
        </p>
      </Card>

      <Card title="An LLM as the controller" color="var(--color-policy)">
        <p>
          Classic RL learns π by gradient updates from reward. Here π is a pretrained model used <i>in context</i>: the prompt encodes
          oₜ and the history, and the reply is decoded into one action. Nothing is trained while it acts, yet it behaves like a strong
          policy because of its priors about web pages.
        </p>
        <p>
          Compare it with the <b>Random</b> baseline in the Benchmark Runner. With sparse rewards a random policy almost never reaches
          the goal on multi-step tasks, which is why pure RL from scratch on the web is so sample-hungry.
        </p>
      </Card>

      <Card title="Trajectories → behavior cloning" color="var(--color-reward)">
        <p>
          Behavior cloning treats the teacher's actions as labels: maximize Σ log π<sub>θ</sub>(aₜ* | oₜ) over recorded steps. Claude
          plays the teacher; a smaller or cheaper student is fine-tuned on the dataset. Because we save the <i>exact prompt</i> at each
          step, the SFT rows match what the policy really conditioned on.
        </p>
        <p>
          <b>Filtered BC</b> keeps only successful episodes, so the reward acts as a quality filter (rejection sampling). Running many
          seeds at a non-zero temperature, then filtering, is a simple and effective recipe.
        </p>
      </Card>

      <Card title="Distribution shift and DAgger" color="var(--color-act)">
        <p>
          A cloned policy makes small mistakes that take it to states the teacher never visited, and errors compound. DAgger fixes this
          by letting the learner act and asking the expert what it <i>should</i> have done in those states.
        </p>
        <p>
          Turn on <b>Approve each action</b> and edit a proposal: the step is recorded with a human label, and the rejected proposal is
          kept. That gives you corrective data on exactly the states where the policy goes wrong, plus preference pairs for DPO.
        </p>
      </Card>

      <Panel className="lg:col-span-2" title="Try it in five minutes" accent="var(--color-env)" bodyClass="p-3">
        <ol className="grid gap-2 text-sm md:grid-cols-2 xl:grid-cols-5">
          <TryStep n={1} tab="live" setTab={setTab}>
            Reset <code>miniwob.click-button</code>, press <b>Propose</b>, read the prompt and the response tabs.
          </TryStep>
          <TryStep n={2} tab="live" setTab={setTab}>
            Approve, and watch the loop diagram move through observe → think → act.
          </TryStep>
          <TryStep n={3} tab="live" setTab={setTab}>
            On a new seed, change Claude's action before approving. That step is now a human correction.
          </TryStep>
          <TryStep n={4} tab="bench" setTab={setTab}>
            Run the MiniWoB starter set with Claude, then with Random. Compare success rates.
          </TryStep>
          <TryStep n={5} tab="data" setTab={setTab}>
            Build an SFT dataset from successful episodes and download the JSONL.
          </TryStep>
        </ol>
      </Panel>
    </div>
  );
}

function Card({ title, color, children }: { title: string; color: string; children: ReactNode }) {
  return (
    <Panel title={title} accent={color} bodyClass="space-y-2.5 p-3 text-[13.5px] leading-relaxed text-ink-300 [&_b]:text-ink-100 [&_code]:font-mono [&_code]:text-[12px] [&_code]:text-obs">
      {children}
    </Panel>
  );
}

function TryStep({ n, tab, setTab, children }: { n: number; tab: Tab; setTab: (t: Tab) => void; children: ReactNode }) {
  return (
    <li>
      <button
        onClick={() => setTab(tab)}
        className="h-full w-full rounded-md border border-ink-700 p-2.5 text-left text-ink-300 transition hover:border-env [&_b]:text-ink-100 [&_code]:font-mono [&_code]:text-[11px] [&_code]:text-obs"
      >
        <span className="mb-1 block font-mono text-xs text-env">{n}</span>
        {children}
      </button>
    </li>
  );
}
