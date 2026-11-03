/**
 * Zod schemas for everything the dashboard receives or sends.
 * REST responses and WebSocket events are parsed before they reach the store,
 * and the new-run form is validated with `NewRun` before it is posted.
 */
import { z } from "zod";

export const Difficulty = z.object({
  n_dates: z.number().int().min(1).max(14),
  n_origins: z.number().int().min(1).max(3),
  pages: z.number().int().min(1).max(8),
  page_size: z.number().int().min(3).max(20),
  hidden_fees: z.boolean(),
  sort_available: z.boolean(),
  passengers: z.number().int().min(1).max(4),
  bags: z.boolean(),
  require_nonstop: z.boolean(),
  depart_after: z.boolean(),
  require_refundable: z.boolean(),
  decoys: z.number().int().min(0).max(20),
});
export type Difficulty = z.infer<typeof Difficulty>;

export const Policy = z.enum(["scripted", "claude", "openai"]);
export type Policy = z.infer<typeof Policy>;

export const Config = z.object({
  presets: z.record(z.string(), Difficulty),
  models: z.array(z.string()),
  default_model: z.string(),
  has_api_key: z.boolean(),
  site_url: z.string(),
  site_up: z.boolean(),
});
export type Config = z.infer<typeof Config>;

export const NewRun = z.object({
  difficulty: Difficulty,
  seed: z.number().int().min(0),
  policy: Policy,
  model: z.string().nullable(),
  max_steps: z.number().int().min(1).max(1000),
  headless: z.boolean(),
});
export type NewRun = z.infer<typeof NewRun>;

const Constraint = z.object({
  name: z.string(),
  ok: z.boolean(),
  expected: z.union([z.string(), z.number()]),
  actual: z.union([z.string(), z.number()]),
});

export const Outcome = z.object({
  booked: z.boolean(),
  success: z.boolean(),
  reward: z.number(),
  reason: z.string(),
  constraints: z.array(Constraint),
  constraints_ok: z.boolean().optional(),
  booked_flight_id: z.number().optional(),
  booked_total: z.number().optional(),
  true_total_for_task: z.number().optional(),
  optimal_flight_id: z.number(),
  optimal_total: z.number(),
  regret: z.number().nullable().optional(),
});
export type Outcome = z.infer<typeof Outcome>;

export const RunStatus = z.enum(["queued", "running", "done", "error", "stopped"]);

export const Run = z.object({
  id: z.string(),
  episode_id: z.string().nullable(),
  difficulty: Difficulty.passthrough(),
  seed: z.number(),
  policy: Policy,
  model: z.string().nullable(),
  max_steps: z.number(),
  status: RunStatus,
  reward: z.number().nullable(),
  success: z.number().nullable(),
  n_steps: z.number(),
  input_tokens: z.number(),
  output_tokens: z.number(),
  outcome: Outcome.nullable(),
  error: z.string().nullable(),
  started_at: z.number().nullable(),
  ended_at: z.number().nullable(),
});
export type Run = z.infer<typeof Run>;

export const PageType = z.enum(["search", "results", "details", "review", "confirmation", "other"]);
export type PageType = z.infer<typeof PageType>;

export const Step = z.object({
  run_id: z.string(),
  t: z.number().int(),
  url: z.string(),
  page: PageType,
  thought: z.string().nullable(),
  notes: z.string().nullable(),
  action: z.string(),
  error: z.string().nullable(),
  prompt_tokens: z.number().nullable(),
  output_tokens: z.number().nullable(),
  latency_ms: z.number().nullable(),
  axtree: z.string().nullable().optional(),
});
export type Step = z.infer<typeof Step>;

export const RunDetail = z.object({ run: Run, steps: z.array(Step) });

const Cell = z.object({
  origin: z.string(),
  date: z.string(),
  n_pages: z.number(),
  pages_seen: z.array(z.number()),
  n_flights: z.number(),
  n_valid: z.number(),
  best_valid_total: z.number().nullable(),
  has_optimum: z.boolean(),
  flights_seen: z.number(),
});
export type Cell = z.infer<typeof Cell>;

const LedgerEntry = z.object({
  id: z.number(),
  flight_no: z.string(),
  airline: z.string(),
  origin: z.string(),
  date: z.string(),
  depart: z.string(),
  stops: z.number(),
  cabin: z.string(),
  refundable: z.boolean(),
  listed_fare: z.number(),
  true_total: z.number(),
  kind: z.string(),
  violations: z.array(z.string()),
  page: z.number(),
  seen: z.boolean(),
  opened: z.boolean(),
  reviewed: z.boolean(),
  booked: z.boolean(),
  first_seen_step: z.number().nullable(),
});
export type LedgerEntry = z.infer<typeof LedgerEntry>;

export const Task = z.object({
  origins: z.array(z.string()),
  origin_metro: z.string(),
  dest: z.string(),
  dates: z.array(z.string()),
  passengers: z.number(),
  bags: z.number(),
  nonstop: z.boolean(),
  cabin: z.string(),
  refundable: z.boolean(),
  depart_after: z.string().nullable(),
  goal: z.string(),
});

export const Coverage = z.object({
  episode_id: z.string(),
  task: Task,
  optimum: z.object({ flight_id: z.number(), total: z.number(), origin: z.string(), date: z.string(), page: z.number() }).passthrough(),
  cells: z.array(Cell),
  ledger: z.array(LedgerEntry),
  diagnosis: z.string(),
  summary: z.object({
    searches_needed: z.number(),
    searches_done: z.number(),
    pages_total: z.number(),
    pages_seen: z.number(),
    flights_seen: z.number(),
    flights_total: z.number(),
    details_opened: z.number(),
    reviews: z.number(),
    valid_seen: z.number(),
    valid_total: z.number(),
  }),
});
export type Coverage = z.infer<typeof Coverage>;

export const ServerEvent = z.discriminatedUnion("type", [
  z.object({ type: z.literal("run"), run: Run }),
  z.object({ type: z.literal("step"), run_id: z.string(), step: Step }),
]);
export type ServerEvent = z.infer<typeof ServerEvent>;
