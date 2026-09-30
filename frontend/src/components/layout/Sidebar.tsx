import type { Status } from "../../api/catalog";
import { getStatus } from "../../api/catalog";
import { useApi } from "../../api/useApi";
import { int } from "../../format";
import { Card } from "../ui/Card";
import { ErrorBox, Loading } from "../ui/Feedback";
import { InfoTooltip } from "../ui/InfoTooltip";
import { Metric } from "../ui/Metric";

const ROLE_LABEL: Record<string, string> = {
  champion: "Đang dùng để chấm điểm",
  challenger: "Bản thử - chưa qua holdout",
};

export function Sidebar({ refreshKey }: { refreshKey: number }) {
  const { data, error } = useApi(getStatus, [refreshKey]);
  return (
    <aside className="sidebar">
      {error && <ErrorBox message={error} />}
      {!error && !data && <Loading />}
      {data && <SidebarBody status={data} />}
    </aside>
  );
}

function SidebarBody({ status }: { status: Status }) {
  if (!status.db_ready || !status.counts) {
    return (
      <Card kind="danger">
        <b>PostgreSQL chưa sẵn sàng</b>
        <pre>docker compose -f infra/docker-compose.yml up -d</pre>
        <small>{status.message.slice(0, 160)}</small>
      </Card>
    );
  }
  const m = status.active_model;
  const c = status.counts;
  return (
    <>
      {m ? (
        <Card kind="ok">
          <div className="card-title-row">
            <b>Model đang dùng</b>
            <InfoTooltip>
              <p>
                Dữ liệu: Home Credit Default Risk (Kaggle) · bộ feature <code>{status.feature_set_version}</code>{" "}
                ({status.n_features} cột).
              </p>
              <p>Kiến trúc: ensemble stacking gồm 5 model con.</p>
              {m.auc != null && (
                <p>
                  AUC {m.auc.toFixed(2)} đo trên tập holdout (hồ sơ chưa từng dùng để huấn luyện), chấm một lần duy
                  nhất.
                </p>
              )}
            </InfoTooltip>
          </div>
          {m.model_name} v{m.version}
          <br />
          {ROLE_LABEL[m.role] ?? m.role}
        </Card>
      ) : (
        <Card kind="warn">
          Chưa kích hoạt model nào. Sang tab <b>Train lại</b>.
        </Card>
      )}
      <div className="metrics vertical">
        <Metric label="Hồ sơ" value={int(c.applicants)} />
        <Metric label="Đã biết kết quả" value={int(c.labeled)} />
        <Metric label="Lần chấm điểm" value={int(c.predictions)} />
        <Metric label="Bản model" value={int(c.model_versions)} />
      </div>
    </>
  );
}
