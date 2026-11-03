/**
 * Zod schemas: the single source of truth for every piece of data that crosses
 * the WebSocket, and for the settings the UI is allowed to hold.
 *
 * - Server events are parsed with `ServerEvent.safeParse` before they touch the
 *   store, so a malformed message can never corrupt UI state.
 * - Client commands are built through `ClientCommand.parse`, so the UI cannot
 *   send something the Python side (server/protocol.py) would reject.
 * - Settings updates go through `Settings.safeParse`, so the store only ever
 *   holds valid, clamped values. That is the "Zod as state controller" part.
 */
import { z } from "zod";

export const PolicyName = z.enum(["claude", "random", "human"]);
export type PolicyName = z.infer<typeof PolicyName>;

export const Phase = z.enum([
  "idle",
  "resetting",
  "observing",
  "thinking",
  "awaiting_approval",
  "acting",
  "done",
]);
export type Phase = z.infer<typeof Phase>;

// ------------------------------------------------------------------ records

export const Observation = z.object({
  t: z.number().int(),
  goal: z.string(),
  url: z.string(),
  open_tabs: z.array(z.string()).default([]),
  axtree: z.string(),
  screenshot: z.string().nullable().optional(),
  last_action: z.string().default(""),
  last_action_error: z.string().default(""),
});
export type Observation = z.infer<typeof Observation>;

export const Prompt = z.object({ system: z.string(), user: z.string() });
export type Prompt = z.infer<typeof Prompt>;

export const Proposal = z.object({
  t: z.number().int(),
  policy: PolicyName,
  model: z.string().nullable().optional(),
  thought: z.string(),
  action: z.string(),
  raw: z.string().default(""),
  prompt: Prompt.nullable().optional(),
  latency_ms: z.number().default(0),
  usage: z.object({ input_tokens: z.number(), output_tokens: z.number() }).default({
    input_tokens: 0,
    output_tokens: 0,
  }),
});
export type Proposal = z.infer<typeof Proposal>;

export const Step = z.object({
  t: z.number().int(),
  obs: Observation,
  proposal: Proposal.nullable().optional(),
  action: z.string(),
  action_source: PolicyName,
  prompt: Prompt.nullable().optional(),
  thought: z.string().default(""),
  reward: z.number(),
  terminated: z.boolean(),
  truncated: z.boolean(),
  action_error: z.string().default(""),
  env_ms: z.number().default(0),
});
export type Step = z.infer<typeof Step>;

export const EpisodeSummary = z.object({
  id: z.string(),
  task: z.string(),
  seed: z.number().int(),
  policy: PolicyName,
  model: z.string().nullable().optional(),
  started_at: z.number(),
  ended_at: z.number().nullable().optional(),
  n_steps: z.number().int(),
  total_reward: z.number(),
  success: z.boolean(),
  terminated: z.boolean(),
  truncated: z.boolean(),
  human_interventions: z.number().int().default(0),
  goal: z.string().optional(),
});
export type EpisodeSummary = z.infer<typeof EpisodeSummary>;

export const Episode = EpisodeSummary.extend({
  goal: z.string().default(""),
  steps: z.array(Step).default([]),
});
export type Episode = z.infer<typeof Episode>;

export const Config = z.object({
  has_api_key: z.boolean(),
  default_model: z.string(),
  models: z.array(z.string()),
  benchmarks: z.array(z.string()),
  action_space: z.string(),
  system_prompt: z.string(),
});
export type Config = z.infer<typeof Config>;

export const TaskCatalog = z.record(z.string(), z.array(z.string()));

// ------------------------------------------------------------------ server -> client

export const ServerEvent = z.discriminatedUnion("type", [
  z.object({ type: z.literal("hello"), config: Config }),
  z.object({ type: z.literal("phase"), phase: Phase }),
  z.object({ type: z.literal("episode_start"), episode: EpisodeSummary }),
  z.object({ type: z.literal("observation"), obs: Observation.nullable() }),
  z.object({
    type: z.literal("policy_start"),
    t: z.number(),
    policy: PolicyName,
    model: z.string().nullable().optional(),
  }),
  z.object({ type: z.literal("policy_delta"), text: z.string() }),
  z.object({ type: z.literal("proposal"), proposal: Proposal }),
  z.object({ type: z.literal("step"), step: Step }),
  z.object({ type: z.literal("episode_end"), episode: EpisodeSummary }),
  z.object({
    type: z.literal("batch_progress"),
    done: z.number(),
    total: z.number(),
    results: z.array(EpisodeSummary),
    running: z.boolean(),
  }),
  z.object({ type: z.literal("error"), message: z.string() }),
  z.object({ type: z.literal("info"), message: z.string() }),
]);
export type ServerEvent = z.infer<typeof ServerEvent>;

// ------------------------------------------------------------------ client -> server

const Temperature = z.number().min(0).max(1).nullable().optional();

export const ClientCommand = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("reset"),
    task: z.string().startsWith("browsergym/"),
    seed: z.number().int().min(0),
    max_steps: z.number().int().min(1).max(50),
    headless: z.boolean(),
    use_screenshot: z.boolean(),
  }),
  z.object({
    type: z.literal("think"),
    policy: PolicyName.exclude(["human"]),
    model: z.string().nullable().optional(),
    temperature: Temperature,
  }),
  z.object({
    type: z.literal("act"),
    action: z.string().min(1, "Action is empty"),
    source: PolicyName,
    thought: z.string().default(""),
  }),
  z.object({
    type: z.literal("run"),
    policy: PolicyName.exclude(["human"]),
    model: z.string().nullable().optional(),
    temperature: Temperature,
    supervised: z.boolean(),
  }),
  z.object({ type: z.literal("pause") }),
  z.object({
    type: z.literal("batch"),
    tasks: z.array(z.string()).min(1, "Pick at least one task"),
    seeds: z.array(z.number().int()).min(1),
    policy: PolicyName.exclude(["human"]),
    model: z.string().nullable().optional(),
    temperature: Temperature,
    max_steps: z.number().int().min(1).max(50),
    use_screenshot: z.boolean(),
  }),
  z.object({ type: z.literal("cancel") }),
]);
export type ClientCommand = z.input<typeof ClientCommand>;

// ------------------------------------------------------------------ UI settings

export const Settings = z.object({
  task: z.string().default("browsergym/miniwob.click-button"),
  seed: z.number().int().min(0).max(1_000_000).default(0),
  maxSteps: z.number().int().min(1).max(50).default(10),
  policy: PolicyName.default("claude"),
  model: z.string().nullable().default(null),
  temperature: z.number().min(0).max(1).nullable().default(null),
  supervised: z.boolean().default(true),
  headless: z.boolean().default(true),
  useScreenshot: z.boolean().default(false),
  gamma: z.number().min(0).max(1).default(0.95),
});
export type Settings = z.infer<typeof Settings>;

// ------------------------------------------------------------------ BC dataset rows

export const BcFormat = z.enum(["sft_chat", "obs_action", "preference"]);
export type BcFormat = z.infer<typeof BcFormat>;

export const SftChatRow = z.object({
  messages: z.array(
    z.object({ role: z.enum(["system", "user", "assistant"]), content: z.string() }),
  ),
  metadata: z.object({
    episode_id: z.string(),
    task: z.string(),
    t: z.number(),
    action_source: PolicyName,
    return_to_go: z.number(),
    episode_success: z.boolean(),
  }),
});

export const ObsActionRow = z.object({
  episode_id: z.string(),
  task: z.string(),
  t: z.number(),
  goal: z.string(),
  observation: z.string(),
  action: z.string(),
  reward: z.number(),
  return_to_go: z.number(),
  done: z.boolean(),
});

export const PreferenceRow = z.object({
  prompt: z.array(z.object({ role: z.enum(["system", "user"]), content: z.string() })),
  chosen: z.string(),
  rejected: z.string(),
  metadata: z.object({ episode_id: z.string(), task: z.string(), t: z.number() }),
});
