import { useState } from "react";

import { errorMessage } from "../api/client";
import type { RetrainStatus } from "../api/retrain";
import {
  getRetrainStatus, promote, registerChampion, revealLabels, runRetrain, setLabel,
} from "../api/retrain";
import { useApi } from "../api/useApi";
import { Card, type CardKind } from "../components/ui/Card";
import { Caveat } from "../components/ui/Caveat";
import { DataTable } from "../components/ui/DataTable";
import { ErrorBox, Loading } from "../components/ui/Feedback";
import { Metric } from "../components/ui/Metric";
import { int, pct } from "../format";
import type { PageProps, Row } from "../types/common";

const REVEAL_CAVEAT =
  "Home Credit là lát cắt tĩnh nên một phần nhãn được giấu sẵn lúc seed rồi tiết lộ dần ở đây - đó " +
  "là thứ làm cho vòng đời train lại có ý nghĩa. Nhãn tiết lộ là nhãn thật từ dữ liệu gốc, không " +
  "sinh ngẫu nhiên; ta chỉ chọn khi nào cho biết.";

const CHALLENGER_CAVEAT =
  "Bản train lại là challenger: LightGBM đơn, train trên phần hồ sơ đã biết kết quả trong DB. Nó " +
  "chưa qua holdout, nên AUC của nó không so được với bảng (holdout 0.7875 của ensemble " +
  "stacking). Muốn thay model cuối thì phải chạy lại quy trình offline đầy đủ.";

type Message = { kind: CardKind; text: string };
type Busy = "reveal" | "label" | "run" | "registry" | null;

export default function RetrainPage({ onChanged }: PageProps) {
  const status = useApi(getRetrainStatus, []);
  const [busy, setBusy] = useState<Busy>(null);
  const [message, setMessage] = useState<Message | null>(null);

  async function act(name: Exclude<Busy, null>, call: () => Promise<Message>) {
    setBusy(name);
    setMessage(null);
    try {
      setMessage(await call());
      status.reload();
      onChanged();
    } catch (e) {
      setMessage({ kind: "danger", text: errorMessage(e) });
    } finally {
      setBusy(null);
    }
  }

  if (status.error) return <ErrorBox message={status.error} />;
  if (!status.data) return <Loading />;
  const s = status.data;

  return (
    <section>
      <h2>Vòng đời train lại</h2>
      {message && <Card kind={message.kind}>{message.text}</Card>}
      <ProgressBlock progress={s.progress} />
      <RevealBlock
        remaining={s.available_to_reveal}
        busy={busy}
        onReveal={(n) =>
          act("reveal", async () => {
            const r = await revealLabels(n);
            return {
              kind: "ok",
              text: `Đã tiết lộ ${int(r.n_revealed)} hồ sơ (${int(r.n_default)} ca vỡ nợ, ${pct(r.n_default / Math.max(r.n_revealed, 1), 1)}). Còn ${int(r.n_remaining)}.`,
            };
          })
        }
      />
      <PendingBlock
        pending={s.pending}
        busy={busy}
        onSave={(id, actual) =>
          act("label", async () => {
            await setLabel(id, actual);
            return { kind: "ok", text: `Đã ghi kết quả cho bản ghi #${id}` };
          })
        }
      />
      <RetrainBlock
        s={s}
        busy={busy}
        onRun={(activate) =>
          act("run", async () => {
            const r = await runRetrain(activate);
            const text =
              `Xong: ${r.model_name} v${r.version} - AUC valid ${r.auc_valid.toFixed(4)}, KS ${r.ks_valid.toFixed(4)}, ` +
              `train ${int(r.n_train)} dòng trong ${r.fit_seconds.toFixed(1)}s.`;
            return r.overfit_warning
              ? {
                kind: "warn",
                text: `${text} AUC train ${r.auc_train.toFixed(4)} cao hơn valid ${r.auc_valid.toFixed(4)} khá nhiều - dấu hiệu khớp quá mức do tập train nhỏ.`,
              }
              : { kind: "ok", text };
          })
        }
      />
      <RegistryBlock
        models={s.models}
        busy={busy}
        onRegister={() =>
          act("registry", async () => {
            const m = await registerChampion();
            return { kind: "ok", text: `Đã đăng ký ${m.model_name} v${m.version} làm model đang dùng.` };
          })
        }
        onActivate={(id) =>
          act("registry", async () => {
            const m = await promote(id);
            return { kind: "ok", text: `Đã kích hoạt bản #${m.id} (${m.model_name} v${m.version}).` };
          })
        }
      />
    </section>
  );
}

function ProgressBlock({ progress }: { progress: RetrainStatus["progress"] }) {
  const ratio = progress.labeled / Math.max(progress.total, 1);
  return (
    <>
      <div className="metrics">
        <Metric label="Hồ sơ trong hệ thống" value={int(progress.total)} />
        <Metric label="Đã biết kết quả" value={int(progress.labeled)} />
        <Metric label="Còn chờ kết quả" value={int(progress.unlabeled)} />
      </div>
      <progress value={progress.labeled} max={Math.max(progress.total, 1)} />
      <p className="muted">{pct(ratio, 1)} hồ sơ đã có nhãn thật</p>
    </>
  );
}

type RevealProps = { remaining: number; busy: Busy; onReveal: (n: number) => void };

function RevealBlock({ remaining, busy, onReveal }: RevealProps) {
  const [n, setN] = useState(1000);
  return (
    <div className="panel">
      <div className="card-title-row">
        <h3>Nhãn thật về theo thời gian</h3>
        <Caveat>{REVEAL_CAVEAT}</Caveat>
      </div>
      <p className="muted">Còn {int(remaining)} hồ sơ đang giấu nhãn.</p>
      <div className="row">
        <label>
          Số hồ sơ tiết lộ{" "}
          <input type="number" min={100} max={5000} step={100} value={n} onChange={(e) => setN(Number(e.target.value))} />
        </label>
      </div>
      <div className="action-end">
        <button disabled={busy !== null || remaining === 0 || n < 100 || n > 5000} onClick={() => onReveal(n)}>
          {busy === "reveal" ? "Đang tiết lộ…" : "Tiết lộ nhãn"}
        </button>
      </div>
    </div>
  );
}

type PendingProps = { pending: Row[]; busy: Busy; onSave: (id: number, actual: 0 | 1) => void };

function PendingBlock({ pending, busy, onSave }: PendingProps) {
  const [pick, setPick] = useState<number | null>(null);
  const [actual, setActual] = useState<0 | 1>(0);
  const ids = pending.map((r) => Number(r.id));
  const current = pick !== null && ids.includes(pick) ? pick : ids[0];
  return (
    <div className="panel">
      <h3>Nhập kết quả thật cho hồ sơ đã chấm</h3>
      {pending.length === 0 ? (
        <p className="muted">Không có dự đoán nào đang chờ kết quả.</p>
      ) : (
        <>
          <DataTable 
            rows={pending} 
            columns={[
              { key: "id", label: "Bản ghi #" },
              { key: "sk_id_curr", label: "Mã hồ sơ" },
              { key: "pd", label: "Xác suất vỡ nợ (PD)" },
              { key: "du_doan", label: "Dự đoán" },
              { key: "luc", label: "Thời gian" },
            ]}
          />
          <div className="row">
            <select value={current} onChange={(e) => setPick(Number(e.target.value))}>
              {ids.map((id) => (
                <option key={id} value={id}>
                  Bản ghi #{id}
                </option>
              ))}
            </select>
            <label>
              <input type="radio" checked={actual === 0} onChange={() => setActual(0)} /> Trả được nợ
            </label>
            <label>
              <input type="radio" checked={actual === 1} onChange={() => setActual(1)} /> Vỡ nợ
            </label>
          </div>
          <div className="action-end">
            <button disabled={busy !== null} onClick={() => onSave(current, actual)}>
              {busy === "label" ? "Đang lưu…" : "Lưu"}
            </button>
          </div>
        </>
      )}
    </div>
  );
}

type RetrainProps = { s: RetrainStatus; busy: Busy; onRun: (activate: boolean) => void };

function RetrainBlock({ s, busy, onRun }: RetrainProps) {
  const [activate, setActivate] = useState(false);
  return (
    <div className="panel">
      <div className="card-title-row">
        <h3>Train lại model</h3>
        <Caveat>{CHALLENGER_CAVEAT}</Caveat>
      </div>
      <Card kind={s.can_retrain ? "ok" : "warn"}>{s.reason}</Card>
      <p className="muted">
        Điều kiện tối thiểu: {int(s.min_training_rows)} hồ sơ có nhãn và {int(s.min_positives)} ca vỡ nợ.
      </p>
      <div className="row">
        <label title="Mặc định KHÔNG bật: bản challenger chưa qua holdout.">
          <input type="checkbox" checked={activate} onChange={(e) => setActivate(e.target.checked)} /> Kích
          hoạt luôn bản mới sau khi train
        </label>
      </div>
      <div className="action-end">
        <button disabled={!s.can_retrain || busy !== null} onClick={() => onRun(activate)}>
          {busy === "run" ? "Đang train LightGBM trên dữ liệu trong DB…" : "Train lại ngay"}
        </button>
      </div>
    </div>
  );
}

type RegistryProps = { models: Row[]; busy: Busy; onRegister: () => void; onActivate: (id: number) => void };

function RegistryBlock({ models, busy, onRegister, onActivate }: RegistryProps) {
  const [pick, setPick] = useState<number | null>(null);
  if (models.length === 0) {
    return (
      <div className="panel">
        <h3>Sổ đăng ký model</h3>
        <p className="muted">Chưa có bản model nào.</p>
        <div className="action-end">
          <button disabled={busy !== null} onClick={onRegister}>
            Đăng ký ensemble offline làm model đang dùng
          </button>
        </div>
      </div>
    );
  }
  const ids = models.map((m) => Number(m.id));
  const current = pick !== null && ids.includes(pick) ? pick : ids[0];
  return (
    <div className="panel">
      <h3>Sổ đăng ký model</h3>
      <DataTable 
        rows={models.map(m => ({
          ...m,
          dang_dung: m.dang_dung ? "Đang chạy" : "",
          ghi_chu: typeof m.ghi_chu === 'string' ? m.ghi_chu.replace("", "").replace("", "") : m.ghi_chu
        }))}
        columns={[
          { key: "id", label: "ID" },
          { key: "model", label: "Mô hình" },
          { key: "version", label: "Phiên bản" },
          { key: "vai_tro", label: "Vai trò" },
          { key: "dang_dung", label: "Hoạt động" },
          { key: "auc_train", label: "AUC Train" },
          { key: "auc_valid", label: "AUC Valid" },
          { key: "ks_valid", label: "KS Valid" },
          { key: "n_train", label: "Số lượng Train" },
          { key: "nguong", label: "Ngưỡng" },
          { key: "bo_feature", label: "Bộ Feature" },
          { key: "train_luc", label: "Thời gian Train" },
          { key: "ghi_chu", label: "Ghi chú" },
        ]}
      />
      <div className="row">
        <select value={current} onChange={(e) => setPick(Number(e.target.value))}>
          {models.map((m) => (
            <option key={String(m.id)} value={Number(m.id)}>
              #{String(m.id)} · {String(m.model)} v{String(m.version)} · {String(m.vai_tro)}
            </option>
          ))}
        </select>
      </div>
      <div className="action-end">
        <button disabled={busy !== null} onClick={() => onActivate(current)}>
          {busy === "registry" ? "Đang kích hoạt…" : "Kích hoạt"}
        </button>
      </div>
    </div>
  );
}
