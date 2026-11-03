/**
 * Turning trajectories into behavior-cloning data.
 *
 * Behavior cloning is supervised learning on (observation -> expert action):
 *     maximize  sum_t log pi_theta(a_t* | o_t)
 * Here the "expert" is Claude (or a human correcting Claude), and every row
 * we emit is one (o_t, a_t) pair taken straight from a recorded episode.
 */
import type { BcFormat, Episode, Step } from "./schemas";
import { ObsActionRow, PreferenceRow, SftChatRow } from "./schemas";

export type BcFilters = {
  successOnly: boolean; // "filtered BC": keep only trajectories that earned reward
  dropErrors: boolean; // drop steps whose action raised an error in the env
  includeHuman: boolean; // keep human-chosen actions as labels
};

/** G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... */
export function returnsToGo(steps: Step[], gamma: number): number[] {
  const out = new Array<number>(steps.length).fill(0);
  let g = 0;
  for (let i = steps.length - 1; i >= 0; i--) {
    g = steps[i].reward + gamma * g;
    out[i] = Number(g.toFixed(4));
  }
  return out;
}

export function assistantTarget(step: Step): string {
  const thought = step.thought?.trim();
  return thought
    ? `<thinking>${thought}</thinking>\n<action>${step.action}</action>`
    : `<action>${step.action}</action>`;
}

function keepStep(step: Step, f: BcFilters) {
  if (f.dropErrors && step.action_error) return false;
  if (!f.includeHuman && step.action_source === "human") return false;
  return true;
}

export function toRows(episodes: Episode[], format: BcFormat, f: BcFilters, gamma: number) {
  const rows: unknown[] = [];
  for (const ep of episodes) {
    if (f.successOnly && !ep.success) continue;
    const rtg = returnsToGo(ep.steps, gamma);
    ep.steps.forEach((step, i) => {
      const prompt = step.prompt ?? step.proposal?.prompt;
      if (format === "preference") {
        // A human overriding Claude gives a natural preference pair: chosen vs rejected.
        const p = step.proposal;
        if (!p || !prompt || step.action_source !== "human" || p.action === step.action) return;
        rows.push(
          PreferenceRow.parse({
            prompt: [
              { role: "system", content: prompt.system },
              { role: "user", content: prompt.user },
            ],
            chosen: assistantTarget(step),
            rejected: `<thinking>${p.thought}</thinking>\n<action>${p.action}</action>`,
            metadata: { episode_id: ep.id, task: ep.task, t: step.t },
          }),
        );
        return;
      }
      if (!keepStep(step, f)) return;
      if (format === "sft_chat") {
        if (!prompt) return;
        rows.push(
          SftChatRow.parse({
            messages: [
              { role: "system", content: prompt.system },
              { role: "user", content: prompt.user },
              { role: "assistant", content: assistantTarget(step) },
            ],
            metadata: {
              episode_id: ep.id,
              task: ep.task,
              t: step.t,
              action_source: step.action_source,
              return_to_go: rtg[i],
              episode_success: ep.success,
            },
          }),
        );
      } else {
        rows.push(
          ObsActionRow.parse({
            episode_id: ep.id,
            task: ep.task,
            t: step.t,
            goal: step.obs.goal,
            observation: step.obs.axtree,
            action: step.action,
            reward: step.reward,
            return_to_go: rtg[i],
            done: step.terminated || step.truncated,
          }),
        );
      }
    });
  }
  return rows;
}

export function downloadJsonl(rows: unknown[], filename: string) {
  const blob = new Blob([rows.map((r) => JSON.stringify(r)).join("\n") + "\n"], {
    type: "application/jsonl",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** Extract the bid an action refers to, e.g. click("13") -> "13". */
export function actionBid(action: string | undefined | null): string | null {
  const m = action?.match(/^\s*\w+\(\s*["']([^"']+)["']/);
  return m ? m[1] : null;
}
