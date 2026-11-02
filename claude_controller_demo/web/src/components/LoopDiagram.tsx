import { useLab, useViewedSteps } from "../store/lab";
import { Panel } from "./ui";

type NodeId = "env" | "obs" | "policy" | "action" | "buffer";

const NODES: Record<NodeId, { x: number; y: number; w: number; h: number; color: string; title: string }> = {
  env: { x: 12, y: 96, w: 136, h: 60, color: "var(--color-env)", title: "Environment" },
  obs: { x: 192, y: 10, w: 136, h: 58, color: "var(--color-obs)", title: "Observation oₜ" },
  policy: { x: 372, y: 96, w: 136, h: 60, color: "var(--color-policy)", title: "Policy π" },
  action: { x: 192, y: 184, w: 136, h: 58, color: "var(--color-act)", title: "Action aₜ" },
  buffer: { x: 206, y: 102, w: 108, h: 48, color: "var(--color-reward)", title: "Trajectory τ" },
};

const EDGES: { id: string; d: string; label: string; lx: number; ly: number }[] = [
  { id: "env-obs", d: "M80 96 C80 50, 140 39, 190 39", label: "render AXTree", lx: 14, ly: 66 },
  { id: "obs-policy", d: "M330 39 C380 39, 440 50, 440 94", label: "prompt", lx: 452, ly: 66 },
  { id: "policy-action", d: "M440 158 C440 200, 380 213, 330 213", label: "parse action", lx: 446, ly: 192 },
  { id: "action-env", d: "M190 213 C140 213, 80 200, 80 158", label: "env.step → rₜ", lx: 14, ly: 192 },
];

const ACTIVE: Record<string, { node?: NodeId; edge?: string }> = {
  resetting: { node: "env" },
  observing: { node: "obs", edge: "env-obs" },
  thinking: { node: "policy", edge: "obs-policy" },
  awaiting_approval: { node: "action", edge: "policy-action" },
  acting: { node: "env", edge: "action-env" },
  done: { node: "buffer" },
};

export function LoopDiagram() {
  const phase = useLab((s) => s.phase);
  const obs = useLab((s) => s.obs);
  const steps = useViewedSteps();
  const draft = useLab((s) => s.draft);
  const settings = useLab((s) => s.settings);
  const config = useLab((s) => s.config);
  const pulse = useLab((s) => s.pulse);
  const active = ACTIVE[phase] ?? {};

  const bidCount = obs ? (obs.axtree.match(/\[[^\]]+\]/g) ?? []).length : 0;
  const ret = steps.reduce((g, s) => g + s.reward, 0);
  const last = steps.at(-1);
  const sub: Record<NodeId, string> = {
    env: "Chromium · BrowserGym",
    obs: obs ? `${bidCount} addressable bids` : "AXTree text",
    policy:
      settings.policy === "claude"
        ? (settings.model ?? config?.default_model ?? "Claude")
        : settings.policy === "random"
          ? "random baseline"
          : "you (human)",
    action: draft ? truncate(draft, 20) : last ? truncate(last.action, 20) : "click(\"bid\") …",
    buffer: `${steps.length} steps · G=${ret.toFixed(2)}`,
  };

  return (
    <Panel
      title="The RL loop"
      accent="var(--color-policy)"
      right={<span className="font-mono text-[10px] text-ink-500">t = {obs?.t ?? "–"}</span>}
    >
      <svg viewBox="0 0 520 250" className="w-full" role="img" aria-label="Agent-environment loop">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--color-ink-500)" />
          </marker>
          <marker id="arrow-on" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--color-ink-100)" />
          </marker>
        </defs>

        {EDGES.map((e) => {
          const on = active.edge === e.id;
          return (
            <g key={e.id}>
              <path
                d={e.d}
                fill="none"
                stroke={on ? "var(--color-ink-100)" : "var(--color-ink-700)"}
                strokeWidth={on ? 2 : 1.5}
                className={on ? "flow" : undefined}
                markerEnd={`url(#${on ? "arrow-on" : "arrow"})`}
              />
              <text x={e.lx} y={e.ly} fontSize="9" fill={on ? "var(--color-ink-100)" : "var(--color-ink-500)"} fontFamily="var(--font-mono)">
                {e.label}
              </text>
            </g>
          );
        })}

        {/* every transition is copied into the trajectory buffer */}
        {(["env", "obs", "policy", "action"] as NodeId[]).map((id) => {
          const n = NODES[id];
          return (
            <line
              key={id}
              x1={n.x + n.w / 2}
              y1={n.y + n.h / 2}
              x2={260}
              y2={126}
              stroke="var(--color-reward)"
              strokeOpacity={0.12}
              strokeDasharray="2 4"
            />
          );
        })}

        {(Object.keys(NODES) as NodeId[]).map((id) => {
          const n = NODES[id];
          const on = active.node === id;
          return (
            <g key={id}>
              {id === "buffer" && pulse > 0 && (
                <rect key={pulse} className="ping-once" x={n.x} y={n.y} width={n.w} height={n.h} rx={8} fill="none" stroke={n.color} strokeWidth={2} />
              )}
              <rect
                x={n.x}
                y={n.y}
                width={n.w}
                height={n.h}
                rx={8}
                fill={on ? `color-mix(in srgb, ${n.color} 18%, var(--color-ink-900))` : "var(--color-ink-850)"}
                stroke={on ? n.color : "var(--color-ink-700)"}
                strokeWidth={on ? 2 : 1}
              />
              <text x={n.x + 10} y={n.y + 22} fontSize="12" fontWeight={600} fill={n.color}>
                {n.title}
              </text>
              <text x={n.x + 10} y={n.y + 40} fontSize="10" fill="var(--color-ink-300)" fontFamily="var(--font-mono)">
                {sub[id]}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="px-3 pb-3 text-xs leading-relaxed text-ink-500">
        {phase === "awaiting_approval"
          ? "Claude has proposed aₜ. Approve it, or edit it to inject a human correction before it reaches the environment."
          : phase === "thinking"
            ? "The AXTree text is the state the LLM conditions on. Its reply is decoded into a single action string."
            : phase === "done"
              ? `Episode finished${last ? (last.terminated ? " (terminated by the task)" : " (truncated: step budget)") : ""}. It was saved as a trajectory.`
              : "Each lap around this loop is one transition (oₜ, aₜ, rₜ, oₜ₊₁) recorded into τ."}
      </p>
    </Panel>
  );
}

const truncate = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
