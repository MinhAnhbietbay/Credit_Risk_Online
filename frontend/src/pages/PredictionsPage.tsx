import { getPredictions } from "../api/predictions";
import { useApi } from "../api/useApi";
import { PlotlyChart } from "../components/charts/PlotlyChart";
import { Card } from "../components/ui/Card";
import { Caveat } from "../components/ui/Caveat";
import { DataTable } from "../components/ui/DataTable";
import { ErrorBox, Loading } from "../components/ui/Feedback";
import { Metric } from "../components/ui/Metric";
import { int, pct } from "../format";

export default function PredictionsPage() {
  const { data, error } = useApi(() => getPredictions(1000), []);
  if (error) return <ErrorBox message={error} />;
  if (!data) return <Loading />;
  const { summary } = data;

  if (summary.total === 0) {
    return (
      <section>
        <h2>Dự đoán so với thực tế</h2>
        <Card kind="info">
          Chưa có dự đoán nào. Sang tab <b>Dự đoán</b> chấm vài hồ sơ trước.
        </Card>
      </section>
    );
  }

  return (
    <section>
      <h2>Dự đoán so với thực tế</h2>
      <div className="metrics">
        <Metric label="Tổng dự đoán" value={int(summary.total)} />
        <Metric label="Đã có kết quả thật" value={int(summary.n_labeled)} />
        <Metric label="PD trung bình" value={pct(summary.pd_mean)} />
        <Metric label="Tỉ lệ bị chặn" value={pct(summary.flag_rate, 1)} />
      </div>

      {summary.n_labeled === 0 ? (
        <Card kind="info">
          Chưa bản ghi nào có kết quả thật. Sang tab <b>Train lại</b> để nhập kết quả, hoặc tiết lộ nhãn
          rồi chấm lại.
        </Card>
      ) : (
        <>
          <Caveat>
            Các chỉ số dưới đây tính trên những hồ sơ đã được chấm trên web - một mẫu nhỏ và không ngẫu
            nhiên, không phải đánh giá model. Số đánh giá chính thức nằm ở
            outputs/reports/model_comparison.md (holdout.
          </Caveat>
          {data.auc != null ? (
            <div className="metrics">
              <Metric
                label="AUC trên phần đã có kết quả"
                value={data.auc.toFixed(4)}
                help={`Chỉ ${int(summary.n_labeled)} bản ghi - dao động rất mạnh khi mẫu nhỏ.`}
              />
            </div>
          ) : (
            <p className="muted">{data.auc_note}</p>
          )}
          <div className="grid2">
            {data.confusion && <PlotlyChart figure={data.confusion} />}
            {data.distribution && <PlotlyChart figure={data.distribution} />}
          </div>
        </>
      )}

      <div className="panel">
        <h3>Toàn bộ bản ghi</h3>
        <DataTable 
          rows={data.rows} 
          columns={[
            { key: "id", label: "Bản ghi #" },
            { key: "sk_id_curr", label: "Mã hồ sơ" },
            { key: "pd", label: "Xác suất vỡ nợ (PD)" },
            { key: "du_doan", label: "Dự đoán" },
            { key: "thuc_te", label: "Thực tế" },
            { key: "nguong", label: "Ngưỡng" },
            { key: "nguon", label: "Nguồn" },
            { key: "model_version_id", label: "ID Model" },
            { key: "luc", label: "Thời gian" },
          ]}
        />
      </div>
    </section>
  );
}
