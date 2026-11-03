/**
 * The whole dashboard state in one Zustand store. Components select the slice they
 * need (`useGym(s => s.selected)`); nothing is passed down through props.
 */
import { create } from "zustand";
import { api, connectEvents } from "../shared/api";
import { Difficulty, type Config, type Coverage, type NewRun, type Run, type ServerEvent, type Step } from "../shared/schemas";

export type Tab = "trajectory" | "coverage" | "fares";

type GymState = {
  config: Config | null;
  connected: boolean;
  runs: Run[];
  selectedId: string | null;
  steps: Step[];
  coverage: Coverage | null;
  selectedT: number | null;
  stepDetail: Step | null;
  tab: Tab;
  form: NewRun;
  formError: string | null;
  error: string | null;

  init: () => Promise<void>;
  setForm: (patch: Partial<NewRun>) => void;
  setDifficulty: (patch: Partial<Difficulty>) => void;
  applyPreset: (name: string) => void;
  start: () => Promise<void>;
  stop: () => Promise<void>;
  select: (id: string) => Promise<void>;
  selectStep: (t: number | null) => Promise<void>;
  setTab: (t: Tab) => void;
  refreshCoverage: () => Promise<void>;
};

const DEFAULT_FORM: NewRun = {
  difficulty: {
    n_dates: 3, n_origins: 1, pages: 2, page_size: 8, hidden_fees: true, sort_available: true,
    passengers: 2, bags: true, require_nonstop: true, depart_after: true, require_refundable: false, decoys: 4,
  },
  seed: 0,
  policy: "scripted",
  model: null,
  max_steps: 250,
  headless: true,
};

let coverageTimer: ReturnType<typeof setTimeout> | null = null;

export const useGym = create<GymState>()((set, get) => {
  function onEvent(e: ServerEvent) {
    if (e.type === "run") {
      set((s) => {
        const others = s.runs.filter((r) => r.id !== e.run.id);
        return { runs: [e.run, ...others].sort((a, b) => (a.id < b.id ? 1 : -1)) };
      });
      if (e.run.id === get().selectedId && ["done", "error", "stopped"].includes(e.run.status))
        void get().refreshCoverage();
      if (e.run.id === get().selectedId && e.run.episode_id && !get().coverage) void get().refreshCoverage();
    } else {
      // keep the live step counter current for every run, selected or not
      set((s) => ({
        runs: s.runs.map((r) => (r.id === e.run_id && r.n_steps < e.step.t + 1 ? { ...r, n_steps: e.step.t + 1 } : r)),
      }));
      if (e.run_id !== get().selectedId) return;
      set((s) => (s.steps.some((x) => x.t === e.step.t) ? s : { steps: [...s.steps, e.step] }));
      // live coverage, throttled
      if (!coverageTimer)
        coverageTimer = setTimeout(() => {
          coverageTimer = null;
          void get().refreshCoverage();
        }, 1500);
    }
  }

  return {
    config: null,
    connected: false,
    runs: [],
    selectedId: null,
    steps: [],
    coverage: null,
    selectedT: null,
    stepDetail: null,
    tab: "trajectory",
    form: DEFAULT_FORM,
    formError: null,
    error: null,

    async init() {
      connectEvents(onEvent, (connected) => set({ connected }));
      try {
        const [config, runs] = await Promise.all([api.config(), api.runs()]);
        set({ config, runs });
        if (runs[0]) await get().select(runs[0].id);
      } catch (e) {
        set({ error: String(e) });
      }
    },

    setForm(patch) {
      set((s) => ({ form: { ...s.form, ...patch }, formError: null }));
    },

    setDifficulty(patch) {
      const next = Difficulty.safeParse({ ...get().form.difficulty, ...patch });
      if (next.success) set((s) => ({ form: { ...s.form, difficulty: next.data }, formError: null }));
      else set({ formError: `${next.error.issues[0].path.join(".")}: ${next.error.issues[0].message}` });
    },

    applyPreset(name) {
      const p = get().config?.presets[name];
      if (p) set((s) => ({ form: { ...s.form, difficulty: p }, formError: null }));
    },

    async start() {
      try {
        const id = await api.start(get().form);
        await get().select(id);
      } catch (e) {
        set({ formError: e instanceof Error ? e.message : String(e) });
      }
    },

    async stop() {
      const id = get().selectedId;
      if (id) await api.stop(id);
    },

    async select(id) {
      set({ selectedId: id, steps: [], coverage: null, selectedT: null, stepDetail: null });
      try {
        const d = await api.run(id);
        if (get().selectedId !== id) return;
        set((s) => ({ steps: d.steps, runs: s.runs.map((r) => (r.id === id ? d.run : r)) }));
        if (d.run.episode_id) await get().refreshCoverage();
        if (d.steps.length) await get().selectStep(d.steps.length - 1);
      } catch (e) {
        set({ error: String(e) });
      }
    },

    async selectStep(t) {
      const id = get().selectedId;
      set({ selectedT: t, stepDetail: null });
      if (id === null || t === null) return;
      const s = await api.step(id, t).catch(() => null);
      if (get().selectedT === t) set({ stepDetail: s });
    },

    setTab: (tab) => set({ tab }),

    async refreshCoverage() {
      const id = get().selectedId;
      if (!id) return;
      const cov = await api.coverage(id).catch(() => null);
      if (get().selectedId === id && cov) set({ coverage: cov });
    },
  };
});

export const useSelectedRun = () => useGym((s) => s.runs.find((r) => r.id === s.selectedId) ?? null);
