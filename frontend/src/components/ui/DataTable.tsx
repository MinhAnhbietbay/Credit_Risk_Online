import type { Cell, Row } from "../../types/common";

export type Column = { key: string; label: string; format?: (v: Cell) => string };

function show(v: Cell | undefined): string {
  if (v === null || v === undefined) return "-";
  if (typeof v === "number") {
    return Number.isInteger(v)
      ? v.toLocaleString("en-US")
      : v.toLocaleString("en-US", { maximumFractionDigits: 4 });
  }
  return String(v);
}

export function DataTable({ rows, columns }: { rows: Row[]; columns?: Column[] }) {
  if (rows.length === 0) return <p className="muted">Không có dòng nào.</p>;
  const cols: Column[] = columns ?? Object.keys(rows[0]).map((key) => ({ key, label: key }));
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c.key}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c.key}>{c.format ? c.format(r[c.key] ?? null) : show(r[c.key])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
