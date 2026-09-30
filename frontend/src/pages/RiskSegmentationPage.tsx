import { getRiskSegments } from "../api/segmentation";
import { useApi } from "../api/useApi";
import { PlotlyChart } from "../components/charts/PlotlyChart";
import { Card } from "../components/ui/Card";
import { Caveat } from "../components/ui/Caveat";
import { DataTable } from "../components/ui/DataTable";
import { ErrorBox, Loading } from "../components/ui/Feedback";
import { Metric } from "../components/ui/Metric";
import { int, pct } from "../format";
import type { Cell } from "../types/common";

const asPct = (v: Cell) => (typeof v === "number" ? pct(v) : "-");

export default function RiskSegmentationPage() {
  const { data, error } = useApi(getRiskSegments, []);
  if (error) return <ErrorBox message={error} />;
  if (!data) return <Loading />;

  if (data.total === 0) {
    return (
      <section>
        <h2>Phân khúc rủi ro danh mục</h2>
        <Card kind="info">
          Chưa có dự đoán nào để phân khúc. Sang tab <b>Dự đoán</b> chấm vài hồ sơ trước.
        </Card>
      </section>
    );
  }

  return (
    <section>
      <h2>Phân khúc rủi ro danh mục</h2>
      <div className="metrics">
        <Metric label="Hồ sơ đã chấm" value={int(data.total)} />
        <Metric label="Trong đó đã có kết quả thật" value={int(data.n_labeled)} />
        <Metric label="PD trung bình" value={pct(data.pd_mean)} />
      </div>
      <div className="grid2">
        {data.band_chart && <PlotlyChart figure={data.band_chart} />}
        {data.calibration_chart ? (
          <PlotlyChart figure={data.calibration_chart} />
        ) : (
          <Card kind="info">
            Chưa nhóm nào có kết quả thật nên chưa đối chiếu được. Nhập kết quả ở tab <b>Train lại</b>.
          </Card>
        )}
      </div>
      <div className="panel">
        <div className="card-title-row">
          <h3>Chi tiết từng nhóm</h3>
          <Caveat>
            Đây là phân khúc của những hồ sơ đã được chấm trên web, không phải toàn danh mục. Cột "Vỡ nợ
            thật" chỉ tính trên phần đã biết kết quả nên rất nhạy khi mẫu nhỏ - đừng đọc nó như số hiệu
            chỉnh của model. Số hiệu chỉnh chính thức (Brier, đường calibration) nằm ở
            outputs/reports/model_comparison.md và outputs/plots/ph1_calibration.png.
          </Caveat>
        </div>
        <DataTable
          rows={data.bands}
          columns={[
            { key: "nhom", label: "Nhóm" },
            { key: "so_ho_so", label: "Số hồ sơ" },
            { key: "pd_tb", label: "PD trung bình", format: asPct },
            { key: "n_co_nhan", label: "Số có kết quả thật" },
            { key: "ti_le_vo_no_that", label: "Vỡ nợ thật", format: asPct },
          ]}
        />
      </div>
    </section>
  );
}
