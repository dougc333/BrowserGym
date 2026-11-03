/**
 * A tiny WebSocket client with Zod at both edges:
 *   inbound  -> ServerEvent.safeParse   (reject anything malformed)
 *   outbound -> ClientCommand.parse      (never send an invalid command)
 */
import { ClientCommand, ServerEvent } from "./schemas";

type Handlers = {
  onEvent: (e: ServerEvent) => void;
  onStatus: (s: "connecting" | "open" | "closed") => void;
  onProtocolError: (msg: string) => void;
};

let ws: WebSocket | null = null;
let retry: ReturnType<typeof setTimeout> | null = null;

export function connect(h: Handlers) {
  if (ws && ws.readyState <= WebSocket.OPEN) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  h.onStatus("connecting");
  ws.onopen = () => h.onStatus("open");
  ws.onclose = () => {
    h.onStatus("closed");
    ws = null;
    if (retry) clearTimeout(retry);
    retry = setTimeout(() => connect(h), 2000); // the bridge may still be starting
  };
  ws.onmessage = (msg) => {
    let data: unknown;
    try {
      data = JSON.parse(msg.data);
    } catch {
      return h.onProtocolError("Received non-JSON message");
    }
    const parsed = ServerEvent.safeParse(data);
    if (!parsed.success) {
      const issue = parsed.error.issues[0];
      return h.onProtocolError(
        `Dropped invalid '${(data as { type?: string })?.type}' event: ${issue.path.join(".")} ${issue.message}`,
      );
    }
    h.onEvent(parsed.data);
  };
}

/** Returns an error message if the command failed validation or the socket is down. */
export function send(cmd: ClientCommand): string | null {
  const parsed = ClientCommand.safeParse(cmd);
  if (!parsed.success) return parsed.error.issues[0].message;
  if (!ws || ws.readyState !== WebSocket.OPEN) return "Not connected to the bridge server";
  ws.send(JSON.stringify(parsed.data));
  return null;
}
