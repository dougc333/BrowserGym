import { useEffect, useMemo, useState } from "react";
import { useLab } from "../store/lab";
import { downloadJsonl, toRows, type BcFilters } from "../shared/bc";
import type { BcFormat, Episode } from "../shared/schemas";
import { Badge, Button, Field, Panel, Toggle, cx, inputCls, sourceColor } from "./ui";

const FORMATS: { id: BcFormat; label: string; blurb: string }[] = [
  {
    id: "sft_chat",
    label: "SFT chat (LLM policy)",
    blurb:
      "One chat example per step: the exact system + user prompt the policy saw, and the executed action as the assistant turn. Fine-tune any chat model on this to distill Claude's browsing behavior.",
  },
  {
    id: "obs_action",
    label: "(o, a, r, G) tuples",
    blurb:
      "Classic behavior-cloning rows for a non-LLM policy (or a Decision Transformer, which conditions on return-to-go G). The observation is the AXTree text.",
  },
  {
    id: "preference",
    label: "Preference pairs (DPO)",
    blurb:
      "Only steps where a human overrode the policy. The human action is 'chosen', the policy's proposal is 'rejected'. This turns supervision into a reward signal without writing a reward model.",
  },
];

export function DatasetPanel() {
  const episodes = useLab((s) => s.episodes);
  const gamma = useLab((s) => s.settings.gamma);
  const update = useLab((s) => s.updateSettings);
  const { loadEpisodes, fetchEpisode, openReplay, notify } = useLab.getState();

  const [format, setFormat] = useState<BcFormat>("sft_chat");
  const [filters, setFilters] = useState<BcFilters>({ successOnly: true, dropErrors: true, includeHuman: true });
  const [full, setFull] = useState<Record<string, Episode>>({});
  const [loading, setLoading] = useState(false);

  useEffect(() => void loadEpisodes(), [loadEpisodes]);

  // Fetch full episodes (with every step) lazily; summaries are enough for the table.
  useEffect(() => {
    const missing = episodes.filter((e) => !full[e.id]).map((e) => e.id);
    if (!missing.length) return;
    setLoading(true);
    Promise.all(missing.map((id) => fetchEpisode(id).catch(() => null)))
      .then((eps) =>
        setFull((f) => {
          const n = { ...f };
          for (const e of eps) if (e) n[e.id] = e;
          return n;
        }),
      )
      .catch((e) => notify("error", String(e)))
      .finally(() => setLoading(false));
  }, [episodes, full, fetchEpisode, notify]);

  const loaded = useMemo(() => episodes.map((e) => full[e.id]).filter(Boolean) as Episode[], [episodes, full]);
  const rows = useMemo(() => toRows(loaded, format, filters, gamma), [loaded, format, filters, gamma]);
  const nSuccess = episodes.filter((e) => e.success).length;
  const kept = filters.successOnly ? nSuccess : episodes.length;
  const preview = rows[0] ? JSON.stringify(rows[0], clip, 2) : null;

  return (
    <div className="grid gap-3 xl:grid-cols-12">
      <Panel
        className="xl:col-span-5"
        title="Recorded trajectories"
        accent="var(--color-reward)"
        right={
          <Button className="!px-2 !py-0.5 text-xs" onClick={() => void loadEpisodes()}>
            refresh
          </Button>
        }
        bodyClass="flex flex-col"
      >
        <div className="flex gap-4 border-b border-ink-700 px-3 py-2 font-mono text-[11px] text-ink-300">
          <span>{episodes.length} episodes</span>
          <span className="text-reward">{nSuccess} successful</span>
          <span>{episodes.reduce((a, e) => a + e.n_steps, 0)} transitions</span>
        </div>
        <div className="max-h-[560px] overflow-auto scroll-thin">
          <table className="w-full font-mono text-[11px]">
            <tbody>
              {episodes.map((e) => (
                <tr
                  key={e.id}
                  onClick={() => void openReplay(e.id)}
                  className={cx(
                    "cursor-pointer border-b border-ink-800 hover:bg-ink-850",
                    filters.successOnly && !e.success && "opacity-40",
                  )}
                  title="Open in the replay inspector"
                >
                  <td className="py-1.5 pl-3 pr-2">
                    <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: sourceColor(e.policy) }} />
                  </td>
                  <td className="px-2 py-1.5 text-ink-300">{e.task.replace("browsergym/", "")}</td>
                  <td className="px-2 py-1.5 text-ink-500">s{e.seed}</td>
                  <td className="px-2 py-1.5 text-ink-500">{e.n_steps} st</td>
                  <td className="px-2 py-1.5 text-ink-500">{e.human_interventions ? `${e.human_interventions} fix` : ""}</td>
                  <td className="py-1.5 pl-2 pr-3 text-right">
                    <Badge color={e.success ? "var(--color-reward)" : "var(--color-act)"}>{e.success ? "pass" : "fail"}</Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!episodes.length && <p className="p-3 text-sm text-ink-500">Run some episodes first. Each finished episode is saved to <code>runs/</code>.</p>}
        </div>
      </Panel>

      <div className="flex flex-col gap-3 xl:col-span-7">
        <Panel title="Dataset builder" accent="var(--color-policy)" bodyClass="space-y-3 p-3">
          <div className="grid gap-2 md:grid-cols-3">
            {FORMATS.map((f) => (
              <button
                key={f.id}
                onClick={() => setFormat(f.id)}
                className={cx(
                  "rounded-md border p-2.5 text-left transition",
                  format === f.id ? "border-policy bg-policy/10" : "border-ink-700 hover:border-ink-500",
                )}
              >
                <div className="text-sm font-medium">{f.label}</div>
                <div className="mt-1 text-[11px] leading-snug text-ink-500">{f.blurb}</div>
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-end gap-5">
            <Toggle
              checked={filters.successOnly}
              onChange={(v) => setFilters({ ...filters, successOnly: v })}
              label="Successful episodes only"
              hint="Filtered BC / rejection sampling: clone only trajectories that reached the goal."
            />
            <Toggle
              checked={filters.dropErrors}
              onChange={(v) => setFilters({ ...filters, dropErrors: v })}
              label="Drop steps that errored"
            />
            <Toggle
              checked={filters.includeHuman}
              onChange={(v) => setFilters({ ...filters, includeHuman: v })}
              label="Keep human labels"
            />
            <Field label="γ for return-to-go">
              <input
                className={cx(inputCls, "w-20")}
                type="number"
                step={0.01}
                min={0}
                max={1}
                value={gamma}
                onChange={(e) => update({ gamma: Number(e.target.value) })}
              />
            </Field>
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="policy"
              disabled={!rows.length}
              onClick={() => downloadJsonl(rows, `browsergym_${format}_${new Date().toISOString().slice(0, 10)}.jsonl`)}
            >
              Download {rows.length} rows (.jsonl)
            </Button>
            <span className="font-mono text-[11px] text-ink-500">
              {loading ? "loading episodes…" : `from ${kept} of ${episodes.length} episodes`}
            </span>
          </div>
        </Panel>

        <Panel title="One training example" accent="var(--color-obs)" bodyClass="p-0">
          {preview ? (
            <pre className="max-h-[420px] overflow-auto whitespace-pre-wrap p-3 font-mono text-[11px] leading-relaxed text-ink-300 scroll-thin">
              {preview}
            </pre>
          ) : (
            <p className="p-3 text-sm text-ink-500">
              {format === "preference"
                ? "No preference pairs yet. Run an episode with “Approve each action” on and edit one of Claude's proposals."
                : "No rows match these filters yet."}
            </p>
          )}
        </Panel>
      </div>
    </div>
  );
}

// Keep the preview readable: long strings (AXTree, action-space docs) are clipped.
function clip(_key: string, value: unknown) {
  if (typeof value === "string" && value.length > 600) return value.slice(0, 600) + ` …[${value.length - 600} more chars]`;
  return value;
}
