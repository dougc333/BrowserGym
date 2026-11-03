import { isBusy, useLab } from "../store/lab";
import type { PolicyName } from "../shared/schemas";
import { Badge, Button, Field, Toggle, cx, inputCls } from "./ui";

const PHASE_LABEL: Record<string, [string, string]> = {
  idle: ["idle", "var(--color-ink-500)"],
  resetting: ["env.reset()", "var(--color-env)"],
  observing: ["o_t ready", "var(--color-obs)"],
  thinking: ["π(a|o) thinking", "var(--color-policy)"],
  awaiting_approval: ["awaiting approval", "var(--color-act)"],
  acting: ["env.step(a_t)", "var(--color-env)"],
  done: ["episode done", "var(--color-reward)"],
};

export function TaskSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const tasks = useLab((s) => s.tasks);
  const benches = Object.keys(tasks);
  return (
    <select className={cx(inputCls, "w-72")} value={value} onChange={(e) => onChange(e.target.value)}>
      {benches.length === 0 && <option value={value}>{value}</option>}
      {benches.map((b) => (
        <optgroup key={b} label={b}>
          {tasks[b].map((t) => (
            <option key={t} value={t}>
              {t.replace("browsergym/", "")}
            </option>
          ))}
        </optgroup>
      ))}
    </select>
  );
}

export function PolicyPicker({ allowHuman = true }: { allowHuman?: boolean }) {
  const settings = useLab((s) => s.settings);
  const update = useLab((s) => s.updateSettings);
  const config = useLab((s) => s.config);
  const options: { id: PolicyName; label: string; disabled?: boolean }[] = [
    { id: "claude", label: "Claude", disabled: config ? !config.has_api_key : false },
    { id: "random", label: "Random" },
    ...(allowHuman ? [{ id: "human" as const, label: "Human" }] : []),
  ];
  return (
    <div className="flex items-end gap-3">
      <Field label="Policy π">
        <div className="flex overflow-hidden rounded-md border border-ink-700">
          {options.map((o) => (
            <button
              key={o.id}
              disabled={o.disabled}
              title={o.disabled ? "Set ANTHROPIC_API_KEY on the server to enable Claude" : undefined}
              onClick={() => update({ policy: o.id })}
              className={cx(
                "px-3 py-1.5 text-sm transition disabled:opacity-30",
                settings.policy === o.id ? "bg-policy text-ink-950" : "bg-ink-950 text-ink-300 hover:text-ink-100",
              )}
            >
              {o.label}
            </button>
          ))}
        </div>
      </Field>
      {settings.policy === "claude" && config && (
        <>
          <Field label="Model">
            <select
              className={inputCls}
              value={settings.model ?? config.default_model}
              onChange={(e) => update({ model: e.target.value })}
            >
              {config.models.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </Field>
          <Field label="Temp" hint="Sampling temperature. Blank = API default. Higher = more diverse rollouts.">
            <input
              className={cx(inputCls, "w-16")}
              placeholder="def"
              type="number"
              step={0.1}
              min={0}
              max={1}
              value={settings.temperature ?? ""}
              onChange={(e) => update({ temperature: e.target.value === "" ? null : Number(e.target.value) })}
            />
          </Field>
        </>
      )}
    </div>
  );
}

export function EpisodeControls() {
  const settings = useLab((s) => s.settings);
  const update = useLab((s) => s.updateSettings);
  const reset = useLab((s) => s.reset);
  const phase = useLab((s) => s.phase);
  const episode = useLab((s) => s.episode);
  const [label, color] = PHASE_LABEL[phase];

  return (
    <div className="flex flex-wrap items-end gap-x-4 gap-y-3 rounded-lg border border-ink-700 bg-ink-900/90 px-3 py-2.5">
      <Field label="Task (gym env id)">
        <TaskSelect value={settings.task} onChange={(task) => update({ task })} />
      </Field>
      <Field label="Seed" hint="The task generator is seeded: same seed = same instance.">
        <div className="flex gap-1">
          <input
            className={cx(inputCls, "w-20")}
            type="number"
            min={0}
            value={settings.seed}
            onChange={(e) => update({ seed: Number(e.target.value) })}
          />
          <Button onClick={() => update({ seed: Math.floor(Math.random() * 10000) })} title="Random seed">
            ⚄
          </Button>
        </div>
      </Field>
      <Field label="Max steps" hint="Step budget. Hitting it ends the episode as truncated.">
        <input
          className={cx(inputCls, "w-16")}
          type="number"
          min={1}
          max={50}
          value={settings.maxSteps}
          onChange={(e) => update({ maxSteps: Number(e.target.value) })}
        />
      </Field>
      <Button variant="primary" onClick={reset} disabled={isBusy(phase)}>
        env.reset()
      </Button>

      <div className="mx-1 h-9 w-px bg-ink-700" />
      <PolicyPicker />

      <div className="flex flex-col gap-1.5">
        <Toggle
          checked={settings.supervised}
          onChange={(v) => update({ supervised: v })}
          label="Approve each action"
          hint="Supervised autopilot: Claude proposes, you approve or correct (DAgger-style)."
        />
        <Toggle
          checked={settings.useScreenshot}
          onChange={(v) => update({ useScreenshot: v })}
          label="Send screenshot to Claude"
          hint="Adds the page screenshot to the prompt (multimodal observation). Applies on next reset."
        />
        <Toggle
          checked={!settings.headless}
          onChange={(v) => update({ headless: !v })}
          label="Show Chromium window"
          hint="Launch a visible Chromium window on the machine running the bridge. Applies on next reset."
        />
      </div>

      <div className="ml-auto flex flex-col items-end gap-1">
        <Badge color={color}>{label}</Badge>
        {episode && (
          <span className="font-mono text-[11px] text-ink-500">
            {episode.id} · seed {episode.seed}
          </span>
        )}
      </div>
    </div>
  );
}
