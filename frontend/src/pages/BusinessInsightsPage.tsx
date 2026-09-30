import { useState } from "react";

import { getDefaultRates, getModelComparison } from "../api/insights";
import { useApi } from "../api/useApi";
import { PlotlyChart } from "../components/charts/PlotlyChart";
import { Card } from "../components/ui/Card";
import { Caveat } from "../components/ui/Caveat";
import { DataTable } from "../components/ui/DataTable";
import { ErrorBox, Loading } from "../components/ui/Feedback";
import { Metric } from "../components/ui/Metric";
import { int } from "../format";

export default function BusinessInsightsPage() {
  const [picked, setPicked] = useState<string[] | null>(null);
  const rates = useApi(() => getDefaultRates(picked), [picked === null ? null : picked.join(",")]);
  const comparison = useApi(getModelComparison, []);

  if (rates.error) return <ErrorBox message={rates.error} />;
  if (!rates.data) return <Loading />;
  const d = rates.data;

  function toggle(feature: string) {
    const next = d.picked.includes(feature)
      ? d.picked.filter((f) => f !== feature)
      : d.available_features.filter((f) => f === feature || d.picked.includes(f));
    setPicked(next);
  }

  return (
    <section>
      <h2>Insight nghiệp vụ</h2>
      <div className="metrics">
        <Metric label="Hồ sơ trong hệ thống" value={int(d.counts.applicants)} />
        <Metric label="Đã biết kết quả" value={int(d.counts.labeled)} />
        <Metric label="Lần chấm điểm" value={int(d.counts.predictions)} />
      </div>

      {d.counts.labeled === 0 ? (
        <Card kind="info">
          Chưa hồ sơ nào có kết quả thật - sang tab <b>Train lại</b> để tiết lộ nhãn.
        </Card>
      ) : (
        <>
          <div className="panel">
            <h3>Tỉ lệ vỡ nợ thật theo từng yếu tố</h3>
            <p className="muted">
              Mọi con số đo trực tiếp trên hồ sơ đã biết kết quả trong hệ thống - không có khuyến nghị nào
              viết cứng bằng số có sẵn.
            </p>
            <div className="row">
              {d.available_features.map((f) => (
                <label key={f}>
                  <input type="checkbox" checked={d.picked.includes(f)} onChange={() => toggle(f)} /> {f}
                </label>
              ))}
            </div>
            <div className="grid2">
              {d.features.map((f) => (
                <div key={f.feature}>
                  {f.chart ? (
                    <>
                      <PlotlyChart figure={f.chart} />
                      <Card kind="info">{f.sentence}</Card>
                    </>
                  ) : (
                    <p className="muted">{f.feature}: chưa đủ dữ liệu để chia phân vị.</p>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="panel">
            <div className="card-title-row">
              <h3>Yếu tố model coi trọng nhất (SHAP)</h3>
              {comparison.data && comparison.data.shap_ranking.length > 0 && (
                <Caveat>{comparison.data.shap_note}</Caveat>
              )}
            </div>
            {comparison.error && <ErrorBox message={comparison.error} />}
            {comparison.data &&
              (comparison.data.shap_ranking.length === 0 ? (
                <p className="muted">
                  Chưa có outputs/reports/shap_global.csv. Chạy <code>python scripts/ph1_shap.py</code>.
                </p>
              ) : (
                <DataTable 
                  rows={comparison.data.shap_ranking} 
                  columns={[
                    { key: "feature", label: "Yếu tố" },
                    { key: "mean_abs_shap", label: "SHAP trung bình tuyệt đối" },
                    { key: "mo_ta", label: "Mô tả" },
                  ]}
                />
              ))}
          </div>

          <div className="panel">
            <h3>So sánh 6 model (số chính thức)</h3>
            {comparison.data &&
              (comparison.data.model_comparison.length === 0 ? (
                <p className="muted">
                  Chưa có outputs/reports/model_comparison.csv. Chạy <code>python scripts/ph1_report.py</code>.
                </p>
              ) : (
                <>
                  <DataTable 
                    rows={comparison.data.model_comparison} 
                    columns={[
                      { key: "model", label: "Mô hình" },
                      { key: "auc_mean", label: "AUC trung bình" },
                      { key: "auc_std", label: "AUC lệch chuẩn" },
                      { key: "oof_auc", label: "AUC ngoài fold" },
                      { key: "ks", label: "KS" },
                      { key: "threshold", label: "Ngưỡng" },
                      { key: "precision", label: "Precision" },
                      { key: "recall", label: "Recall" },
                      { key: "f1", label: "F1" },
                      { key: "brier", label: "Brier" },
                      { key: "fit_seconds", label: "Thời gian huấn luyện (s)" },
                    ]}
                  />
                  <p className="muted">
                    Số này đo ngoài fold trên 246,008 hồ sơ phần CV; model cuối (ensemble stacking) được chấm
                    trên holdout 61,503 hồ sơ. Không phải số đo trên dữ liệu web.
                  </p>
                </>
              ))}
          </div>
        </>
      )}
    </section>
  );
}
