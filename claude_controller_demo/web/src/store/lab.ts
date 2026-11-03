/**
 * The whole app state lives here, in one Zustand store.
 *
 * Components subscribe to exactly the slice they need with a selector
 * (`useLab(s => s.phase)`), so nothing is passed down through props: the
 * loop diagram, the AXTree panel and the timeline all read the same store.
 *
 * Every write goes through Zod first:
 *   - server events are validated in shared/socket.ts before `applyEvent` runs
 *   - settings changes are validated by `Settings.safeParse` in `updateSettings`
 *   - outgoing commands are validated by `ClientCommand` in shared/socket.ts
 */
import { create } from "zustand";
import * as socket from "../shared/socket";
import {
  Episode,
  EpisodeSummary,
  Settings,
  TaskCatalog,
  type ClientCommand,
  type Config,
  type Observation,
  type Phase,
  type Proposal,
  type ServerEvent,
  type Step,
} from "../shared/schemas";

export type Tab = "live" | "bench" | "data" | "learn";
type LogLine = { id: number; level: "info" | "error"; message: string; ts: number };
type Batch = { done: number; total: number; results: EpisodeSummary[]; running: boolean };

type LabState = {
  connection: "connecting" | "open" | "closed";
  config: Config | null;
  tasks: Record<string, string[]>;
  settings: Settings;
  phase: Phase;
  autopilot: boolean;

  // live episode
  episode: EpisodeSummary | null;
  steps: Step[];
  obs: Observation | null;
  stream: string; // raw policy tokens as they arrive
  proposal: Proposal | null;
  draft: string; // action editor contents (human-in-the-loop)
  pulse: number; // increments on every recorded transition, drives the recorder animation

  // inspection
  tab: Tab;
  selectedStep: number | null;
  replay: Episode | null;

  // benchmark + dataset
  batch: Batch | null;
  episodes: EpisodeSummary[];
  log: LogLine[];

  // actions
  init: () => void;
  updateSettings: (patch: Partial<Settings>) => void;
  setTab: (t: Tab) => void;
  setDraft: (a: string) => void;
  selectStep: (i: number | null) => void;
  reset: () => void;
  think: () => void;
  execute: () => void;
  run: () => void;
  pause: () => void;
  startBatch: (tasks: string[], seeds: number[]) => void;
  cancelBatch: () => void;
  loadEpisodes: () => Promise<void>;
  fetchEpisode: (id: string) => Promise<Episode>;
  openReplay: (id: string) => Promise<void>;
  closeReplay: () => void;
  notify: (level: LogLine["level"], message: string) => void;
};

let logId = 0;

export const useLab = create<LabState>()((set, get) => {
  const send = (cmd: ClientCommand) => {
    const err = socket.send(cmd);
    if (err) get().notify("error", err);
    return !err;
  };
  const policyArgs = () => {
    const s = get().settings;
    return {
      policy: s.policy === "human" ? ("claude" as const) : s.policy,
      model: s.model ?? get().config?.default_model ?? null,
      temperature: s.temperature,
    };
  };

  function applyEvent(e: ServerEvent) {
    switch (e.type) {
      case "hello":
        set({ config: e.config });
        if (!e.config.has_api_key && get().settings.policy === "claude")
          get().updateSettings({ policy: "random" });
        break;
      case "phase":
        set({ phase: e.phase });
        if (e.phase === "done" || e.phase === "idle") set({ autopilot: false });
        break;
      case "episode_start":
        set({
          episode: e.episode,
          steps: [],
          proposal: null,
          stream: "",
          draft: "",
          selectedStep: null,
          replay: null,
        });
        break;
      case "observation":
        set({ obs: e.obs });
        break;
      case "policy_start":
        set({ stream: "", proposal: null });
        break;
      case "policy_delta":
        set((s) => ({ stream: s.stream + e.text }));
        break;
      case "proposal":
        set({ proposal: e.proposal, draft: e.proposal.action });
        break;
      case "step":
        set((s) => ({ steps: [...s.steps, e.step], draft: "", pulse: s.pulse + 1 }));
        break;
      case "episode_end":
        set({ episode: e.episode, autopilot: false });
        get().notify(
          "info",
          `Episode ${e.episode.success ? "succeeded" : "failed"}: ${e.episode.task} (seed ${e.episode.seed}), ${e.episode.n_steps} steps, return ${e.episode.total_reward}`,
        );
        void get().loadEpisodes();
        break;
      case "batch_progress":
        set({ batch: e });
        break;
      case "error":
        set({ autopilot: false });
        get().notify("error", e.message);
        break;
      case "info":
        get().notify("info", e.message);
        break;
    }
  }

  return {
    connection: "connecting",
    config: null,
    tasks: {},
    settings: Settings.parse({}),
    phase: "idle",
    autopilot: false,
    episode: null,
    steps: [],
    obs: null,
    stream: "",
    proposal: null,
    draft: "",
    pulse: 0,
    tab: "live",
    selectedStep: null,
    replay: null,
    batch: null,
    episodes: [],
    log: [],

    init() {
      socket.connect({
        onEvent: applyEvent,
        onStatus: (connection) => set({ connection }),
        onProtocolError: (m) => get().notify("error", m),
      });
      fetch("/api/tasks")
        .then((r) => r.json())
        .then((d) => set({ tasks: TaskCatalog.parse(d) }))
        .catch(() => get().notify("error", "Could not load the task list from /api/tasks"));
      void get().loadEpisodes();
    },

    updateSettings(patch) {
      const next = Settings.safeParse({ ...get().settings, ...patch });
      if (next.success) set({ settings: next.data });
      else get().notify("error", `Invalid setting: ${next.error.issues[0].message}`);
    },

    setTab: (tab) => set({ tab }),
    setDraft: (draft) => set({ draft }),
    selectStep: (selectedStep) => set({ selectedStep }),

    reset() {
      const s = get().settings;
      send({
        type: "reset",
        task: s.task,
        seed: s.seed,
        max_steps: s.maxSteps,
        headless: s.headless,
        use_screenshot: s.useScreenshot,
      });
    },

    think() {
      if (get().settings.policy === "human") {
        get().notify("info", "Human policy: type an action or click a [bid] in the AXTree.");
        return;
      }
      send({ type: "think", ...policyArgs() });
    },

    execute() {
      const { draft, proposal } = get();
      // Unedited proposal = the policy acted. Edited or typed = a human correction.
      const fromPolicy = !!proposal && draft.trim() === proposal.action.trim();
      send({
        type: "act",
        action: draft.trim(),
        source: fromPolicy ? proposal!.policy : "human",
        thought: fromPolicy ? proposal!.thought : "",
      });
    },

    run() {
      if (get().settings.policy === "human") {
        get().notify("info", "Autopilot needs an automatic policy (Claude or random).");
        return;
      }
      if (send({ type: "run", supervised: get().settings.supervised, ...policyArgs() }))
        set({ autopilot: true });
    },

    pause() {
      send({ type: "pause" });
      set({ autopilot: false });
    },

    startBatch(tasks, seeds) {
      const s = get().settings;
      send({
        type: "batch",
        tasks,
        seeds,
        max_steps: s.maxSteps,
        use_screenshot: s.useScreenshot,
        ...policyArgs(),
      });
    },

    cancelBatch: () => void send({ type: "cancel" }),

    async loadEpisodes() {
      try {
        const data = await (await fetch("/api/episodes")).json();
        set({ episodes: EpisodeSummary.array().parse(data) });
      } catch {
        /* bridge not up yet */
      }
    },

    async fetchEpisode(id) {
      const data = await (await fetch(`/api/episodes/${encodeURIComponent(id)}`)).json();
      return Episode.parse(data);
    },

    async openReplay(id) {
      try {
        const ep = await get().fetchEpisode(id);
        set({ replay: ep, selectedStep: 0, tab: "live" });
      } catch (e) {
        get().notify("error", `Could not load episode: ${String(e)}`);
      }
    },

    closeReplay: () => set({ replay: null, selectedStep: null }),

    notify(level, message) {
      set((s) => ({ log: [...s.log.slice(-49), { id: ++logId, level, message, ts: Date.now() }] }));
    },
  };
});

// ------------------------------------------------------------------ derived selectors

/** The steps being viewed: a replayed episode from disk, or the live one. */
export const useViewedSteps = () => useLab((s) => (s.replay ? s.replay.steps : s.steps));

/**
 * The step under inspection. `null` means "the live present": show the current
 * observation and the policy's in-flight proposal.
 */
export const useViewedStep = (): Step | null =>
  useLab((s) => {
    const steps = s.replay ? s.replay.steps : s.steps;
    return s.selectedStep !== null ? (steps[s.selectedStep] ?? null) : null;
  });

export const isBusy = (p: Phase) => p === "resetting" || p === "thinking" || p === "acting";
