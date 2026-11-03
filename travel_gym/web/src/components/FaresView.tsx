import { useGym } from "../store/gym";
import { Card, Pill, cx, money } from "./ui";

/** The cheapest valid flights and every decoy, with what the agent did with each. */
export function FaresView() {
  const cov = useGym((s) => s.coverage);
  const selectStep = useGym((s) => s.selectStep);
  const setTab = useGym((s) => s.setTab);
  if (!cov) return <p className="p-4 text-sm text-slate-500">Available once the episode is generated.</p>;
  const tone = (k: string) => (k === "optimum" ? "green" : k.startsWith("decoy") ? "red" : k === "fee trap" ? "amber" : "slate");

  return (
    <Card title="Fares vs ground truth" className="min-h-0 overflow-auto scroll-thin"
      right={<span className="text-xs text-slate-500">true total = what the task's travelers and bags actually cost</span>}>
      <table className="w-full text-sm">
        <thead className="bg-slate-50 text-left text-[11px] uppercase tracking-wider text-slate-500">
          <tr>
            <th className="px-3 py-2">Kind</th><th className="px-3 py-2">Flight</th><th className="px-3 py-2">Where</th>
            <th className="px-3 py-2 text-right">Listed fare</th><th className="px-3 py-2 text-right">True total</th>
            <th className="px-3 py-2">Breaks</th><th className="px-3 py-2">Agent</th>
          </tr>
        </thead>
        <tbody>
          {cov.ledger.map((f) => (
            <tr key={f.id} className={cx("border-t border-slate-100", f.booked && "bg-sky-50")}>
              <td className="px-3 py-1.5"><Pill tone={tone(f.kind)}>{f.kind}</Pill></td>
              <td className="px-3 py-1.5 font-mono text-xs">{f.flight_no}<div className="font-sans text-[11px] text-slate-500">{f.airline}</div></td>
              <td className="px-3 py-1.5 text-xs">{f.origin} · {f.date} · {f.depart}<div className="text-[11px] text-slate-500">page {f.page} · {f.stops ? `${f.stops} stop` : "nonstop"} · {f.cabin}</div></td>
              <td className="px-3 py-1.5 text-right tabular-nums">{money(f.listed_fare)}</td>
              <td className="px-3 py-1.5 text-right font-medium tabular-nums">{money(f.true_total)}</td>
              <td className="px-3 py-1.5 text-xs text-rose-700">{f.violations.join(", ") || "–"}</td>
              <td className="px-3 py-1.5">
                <div className="flex flex-wrap gap-1">
                  {f.seen ? (
                    <button onClick={() => { setTab("trajectory"); void selectStep(f.first_seen_step ?? 0); }}>
                      <Pill tone="sky">seen t{f.first_seen_step}</Pill>
                    </button>
                  ) : <Pill>not seen</Pill>}
                  {f.opened && <Pill tone="violet">opened</Pill>}
                  {f.reviewed && <Pill tone="amber">total checked</Pill>}
                  {f.booked && <Pill tone={f.kind === "optimum" ? "green" : "red"}>booked</Pill>}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="border-t border-slate-100 px-3 py-2 text-xs text-slate-500">
        Decoys are cheaper but break one requirement. A fee trap has the lowest listed fare among valid flights but a higher
        true total once bags and taxes are added: it is only exposed on the review page.
      </p>
    </Card>
  );
}
