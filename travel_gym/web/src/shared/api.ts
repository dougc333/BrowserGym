/** REST + WebSocket access. Every response is validated with Zod before use. */
import { z } from "zod";
import { Config, Coverage, NewRun, Run, RunDetail, ServerEvent, Step } from "./schemas";

async function get<T>(url: string, schema: z.ZodType<T>): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`);
  return schema.parse(await r.json());
}

export const api = {
  config: () => get("/api/config", Config),
  runs: () => get("/api/runs", z.array(Run)),
  run: (id: string) => get(`/api/runs/${id}`, RunDetail),
  step: (id: string, t: number) => get(`/api/runs/${id}/steps/${t}`, Step),
  coverage: (id: string) => get(`/api/runs/${id}/coverage`, Coverage),
  screenshot: (id: string, t: number) => `/api/screens/${id}/${t}.jpg`,
  async start(body: NewRun): Promise<string> {
    const parsed = NewRun.parse(body);
    const r = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(parsed),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `HTTP ${r.status}`);
    return z.object({ id: z.string() }).parse(await r.json()).id;
  },
  stop: (id: string) => fetch(`/api/runs/${id}/stop`, { method: "POST" }),
};

export function connectEvents(onEvent: (e: ServerEvent) => void, onStatus: (open: boolean) => void) {
  let ws: WebSocket | null = null;
  const open = () => {
    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.onopen = () => onStatus(true);
    ws.onclose = () => {
      onStatus(false);
      setTimeout(open, 2000);
    };
    ws.onmessage = (m) => {
      const parsed = ServerEvent.safeParse(JSON.parse(m.data));
      if (parsed.success) onEvent(parsed.data);
      else console.warn("dropped invalid event", parsed.error.issues[0]);
    };
  };
  open();
}
