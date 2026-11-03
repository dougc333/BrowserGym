import { useGym } from "../store/gym";
import { Card, Pill, Stat, cx, money } from "./ui";

/** Airports × dates: which searches and result pages the agent explored, and where the answer was. */
export function CoverageView() {
  const cov = useGym((s) => s.coverage);
  if (!cov) return <p className="p-4 text-sm text-slate-500">Coverage appears once the episode is generated.</p>;
  const { cells, summary: sm, task } = cov;
  const origins = task.origins;
  const dates = task.dates;
  const cell = (o: string, d: string) => cells.find((c) => c.origin === o && c.date === d)!;
  const fmtDate = (d: string) => new Date(d + "T12:00:00").toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });

  return (
    <div className="space-y-3 overflow-auto scroll-thin">
      <Card>
        <div className="flex flex-wrap gap-x-10 gap-y-3 p-4">
          <Stat label="Searches" value={`${sm.searches_done} / ${sm.searches_needed}`} />
          <Stat label="Result pages" value={`${sm.pages_seen} / ${sm.pages_total}`} />
          <Stat label="Flights seen" value={`${sm.flights_seen} / ${sm.flights_total}`} />
          <Stat label="Valid flights seen" value={`${sm.valid_seen} / ${sm.valid_total}`} />
          <Stat label="Details opened" value={sm.details_opened} />
          <Stat label="True totals checked" value={sm.reviews} />
        </div>
        <p className="border-t border-slate-100 px-4 py-2.5 text-sm">
          <span className="font-semibold">Diagnosis: </span>{cov.diagnosis}
        </p>
      </Card>

      <Card title="Search coverage: allowed airports × dates" right={<span className="text-xs text-slate-500">★ = where the cheapest valid flight is</span>}>
        <div className="overflow-x-auto p-4">
          <table className="border-separate border-spacing-1.5">
            <thead>
              <tr>
                <th />
                {dates.map((d) => <th key={d} className="px-1 text-left text-xs font-medium text-slate-500">{fmtDate(d)}</th>)}
              </tr>
            </thead>
            <tbody>
              {origins.map((o) => (
                <tr key={o}>
                  <th className="pr-2 text-left font-mono text-sm text-slate-700">{o}</th>
                  {dates.map((d) => {
                    const c = cell(o, d);
                    const seen = c.pages_seen.length;
                    return (
                      <td key={d}>
                        <div className={cx("w-36 rounded-lg border p-2",
                          c.has_optimum ? "border-emerald-500 ring-2 ring-emerald-200" : "border-slate-200",
                          seen === 0 && "bg-slate-50")}>
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-slate-500">{c.n_flights} flights</span>
                            {c.has_optimum && <span className="text-emerald-600">★</span>}
                          </div>
                          <div className="mt-1.5 flex gap-1">
                            {Array.from({ length: c.n_pages }, (_, i) => (
                              <span key={i} title={`page ${i + 1}${c.pages_seen.includes(i + 1) ? " (seen)" : ""}`}
                                className={cx("h-2.5 flex-1 rounded-sm", c.pages_seen.includes(i + 1) ? "bg-sky-500" : "bg-slate-200")} />
                            ))}
                          </div>
                          <div className="mt-1.5 text-[11px] text-slate-500">
                            {seen}/{c.n_pages} pages · {c.n_valid} valid
                          </div>
                          <div className="text-xs font-medium tabular-nums">best {money(c.best_valid_total)}</div>
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-2 flex gap-4 text-[11px] text-slate-500">
            <span className="flex items-center gap-1"><span className="h-2 w-4 rounded-sm bg-sky-500" />page seen</span>
            <span className="flex items-center gap-1"><span className="h-2 w-4 rounded-sm bg-slate-200" />page not seen</span>
            <Pill tone="green">optimum: {cov.optimum.origin} · {cov.optimum.date} · page {cov.optimum.page} · {money(cov.optimum.total)}</Pill>
          </div>
        </div>
      </Card>
    </div>
  );
}
