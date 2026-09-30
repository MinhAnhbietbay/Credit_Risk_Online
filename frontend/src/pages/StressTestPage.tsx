import { useState } from "react";

import { errorMessage } from "../api/client";
import type { StressRun } from "../api/stress";
import { getScenarios, runStress } from "../api/stress";
import { useApi } from "../api/useApi";
import { PlotlyChart } from "../components/charts/PlotlyChart";
import { Caveat } from "../components/ui/Caveat";
import { DataTable } from "../components/ui/DataTable";
import { ErrorBox, Loading } from "../components/ui/Feedback";
import { int, pct, signedPct } from "../format";
import type { Cell } from "../types/common";

const num = (v: Cell) => (typeof v === "number" ? v : 0);

export default function StressTestPage() {
  const scenarios = useApi(getScenarios, []);
  const [n, setN] = useState(3000);
  const [result, setResult] = useState<StressRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(await runStress(n));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  if (scenarios.error) return <ErrorBox message={scenarios.error} />;
  if (!scenarios.data) return <Loading />;
  const s = scenarios.data;

  return (
    <section>
      <div className="card-title-row">
        <h2>Stress test danh mục</h2>
        <Caveat>{s.warning}</Caveat>
      </div>
      <label className="row">
        Số hồ sơ lấy mẫu từ danh mục: <b>{int(n)}</b>
        <input
          type="range"
          min={500}
          max={10000}
          step={500}
          value={n}
          onChange={(e) => setN(Number(e.target.value))}
          title="Mẫu càng lớn số càng ổn định, nhưng chạy lâu hơn."
        />
      </label>

      <div className="panel">
        <h3>Kịch bản</h3>
        <DataTable
          rows={s.scenarios}
          columns={[
            { key: "name", label: "Kịch bản" },
            { key: "channel", label: "Kênh" },
            { key: "description", label: "Nghĩa" },
            { key: "shock", label: "Cú sốc" },
            { key: "source", label: "Nguồn con số" },
          ]}
        />
        <div className="action-end">
          <button disabled={busy} onClick={run}>
            {busy ? `Đang chấm lại ${int(n)} hồ sơ × ${s.scenarios.length} kịch bản…` : "Chạy stress test"}
          </button>
        </div>
      </div>
      {error && <ErrorBox message={error} />}

      {result && (
        <>
          <div className="panel">
            <h3>Tác động lên danh mục</h3>
            <DataTable
              rows={result.rows}
              columns={[
                { key: "scenario", label: "Kịch bản" },
                { key: "channel", label: "Kênh" },
                { key: "shock", label: "Cú sốc" },
                { key: "pd_mean", label: "PD trung bình", format: (v) => pct(num(v)) },
                { key: "flag_rate", label: "% bị chặn", format: (v) => pct(num(v), 1) },
                { key: "flag_rate_delta", label: "Chênh so với cơ sở", format: (v) => signedPct(num(v)) },
              ]}
            />
            <PlotlyChart figure={result.band_chart} />
          </div>
          <div className="panel">
            <h3>Đọc kết quả thế nào</h3>
            <p>{s.non_monotonic_note}</p>
            <p className="muted">
              Ngưỡng chặn giữ nguyên {result.threshold.toFixed(4)} ở mọi kịch bản - kịch bản làm dịch chuyển
              điểm số, không dịch chuyển chính sách. Mẫu {int(result.n_sample)} hồ sơ lấy từ DB (không chạm
              holdout).
            </p>
          </div>
        </>
      )}
    </section>
  );
}
