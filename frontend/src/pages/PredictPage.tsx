import { useEffect, useRef, useState } from "react";

import type { ApplicantOption, PredictResult } from "../api/predict";
import { getApplicant, getFormConfig, predictExisting, predictManual } from "../api/predict";
import { errorMessage } from "../api/client";
import { useApi } from "../api/useApi";
import { ApplicantForm } from "../components/ApplicantForm";
import { PlotlyChart } from "../components/charts/PlotlyChart";
import { Badge } from "../components/ui/Badge";
import { Card } from "../components/ui/Card";
import { Caveat } from "../components/ui/Caveat";
import { DataTable } from "../components/ui/DataTable";
import { ErrorBox, Loading } from "../components/ui/Feedback";
import { int, pct } from "../format";
import type { PageProps } from "../types/common";

type Mode = "existing" | "manual";

export default function PredictPage({ onChanged }: PageProps) {
  const config = useApi(getFormConfig, []);
  const [mode, setMode] = useState<Mode>("existing");
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function run(call: () => Promise<PredictResult>) {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await call());
      onChanged();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  function switchMode(m: Mode) {
    setMode(m);
    setResult(null);
    setError(null);
  }

  if (config.error) return <ErrorBox message={config.error} />;
  if (!config.data) return <Loading />;
  const data = config.data;

  return (
    <section>
      <h2>Dự đoán rủi ro vỡ nợ</h2>
      {!data.active_model ? (
        <Card kind="warn">
          Chưa có model nào được kích hoạt. Sang tab <b>Train lại</b> để đăng ký/kích hoạt.
        </Card>
      ) : (
        <>
          <div className="row">
            <label>
              <input type="radio" checked={mode === "existing"} onChange={() => switchMode("existing")} />{" "}
              Hồ sơ có trong hệ thống
            </label>
            <label>
              <input type="radio" checked={mode === "manual"} onChange={() => switchMode("manual")} />{" "}
              Nhập tay
            </label>
          </div>
          {mode === "existing" ? (
            <ExistingApplicant
              applicants={data.applicants}
              busy={busy}
              onScore={(sk) => run(() => predictExisting(sk))}
              onSelectChange={() => {
                setResult(null);
                setError(null);
              }}
            />
          ) : (
            <>
              <p className="muted">Hồ sơ không có trong hệ thống - bản ghi sẽ lưu với sk_id_curr rỗng.</p>
              <ApplicantForm
                fields={data.fields}
                nTotal={data.n_features_total}
                busy={busy}
                onSubmit={(values) => run(() => predictManual(values))}
              />
            </>
          )}
          {error && <ErrorBox message={error} />}
          {result && <PredictResultView r={result} />}
        </>
      )}
    </section>
  );
}

type ExistingProps = {
  applicants: ApplicantOption[];
  busy: boolean;
  onScore: (sk: number) => void;
  onSelectChange: () => void;
};

const SEARCH_RESULTS_LIMIT = 20;

function ExistingApplicant({ applicants, busy, onScore, onSelectChange }: ExistingProps) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [sk, setSk] = useState<number | null>(applicants[0]?.sk_id_curr ?? null);
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onClickOutside(e: MouseEvent) {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, [open]);

  if (sk === null) {
    return (
      <Card kind="warn">
        Không có hồ sơ nào đang chờ kết quả. Chạy <code>python scripts/ph1_db_seed.py</code>.
      </Card>
    );
  }

  const q = query.trim().toLowerCase();
  const filtered = q
    ? applicants.filter((a) => a.label.toLowerCase().includes(q) || String(a.sk_id_curr).includes(q))
    : [];
  const shown = filtered.slice(0, SEARCH_RESULTS_LIMIT);
  const selected = applicants.find((a) => a.sk_id_curr === sk) ?? null;

  function pick(next: ApplicantOption) {
    setSk(next.sk_id_curr);
    setQuery("");
    setOpen(false);
    onSelectChange();
  }

  return (
    <>
      <div className="applicant-search" ref={boxRef}>
        <input
          type="text"
          className="applicant-search-input"
          placeholder="Gõ mã hồ sơ, số tiền vay, tuổi… để tìm"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
        />
        {open && q && (
          <div className="applicant-search-list">
            {shown.length === 0 && <p className="muted">Không tìm thấy hồ sơ khớp.</p>}
            {shown.map((a) => (
              <button
                key={a.sk_id_curr}
                type="button"
                className={`applicant-row${a.sk_id_curr === sk ? " active" : ""}`}
                onClick={() => pick(a)}
              >
                {a.label}
              </button>
            ))}
            {filtered.length > shown.length && (
              <p className="muted applicant-search-more">
                Còn {int(filtered.length - shown.length)} hồ sơ khớp khác - gõ thêm để lọc hẹp lại.
              </p>
            )}
          </div>
        )}
      </div>
      {selected && (
        <div style={{ margin: '1rem 0' }}>
          <p>
            Đã chọn: <b>{selected.label}</b>
          </p>
          <div className="action-end">
            <button disabled={busy} onClick={() => onScore(sk)}>
              {busy ? "Đang chấm…" : "Chấm điểm hồ sơ này"}
            </button>
          </div>
        </div>
      )}
      <ApplicantTopFeatures sk={sk} />
    </>
  );
}

function ApplicantTopFeatures({ sk }: { sk: number }) {
  const { data, error } = useApi(() => getApplicant(sk), [sk]);
  if (error) return <ErrorBox message={error} />;
  if (!data) return <Loading />;
  return (
    <div style={{ marginTop: '1rem' }}>
      <h3 style={{ color: 'var(--text)' }}>{data.top_features.length} yếu tố quan trọng nhất của hồ sơ này</h3>
      <DataTable
        rows={data.top_features}
        columns={[
          { key: "feature", label: "Yếu tố" },
          { key: "value", label: "Giá trị" },
          { key: "mo_ta", label: "Ý nghĩa" },
        ]}
      />
    </div>
  );
}

function PredictResultView({ r }: { r: PredictResult }) {
  return (
    <div className="result">
      <div className="grid2">
        <PlotlyChart figure={r.gauge} />
        <div>
          <Badge pd={r.pd} level={r.risk_level} color={r.risk_color} decision={r.decision} threshold={r.threshold} />
          <p className="muted">
            Model: {r.model_label} · đã lưu bản ghi #{r.prediction_id}
          </p>
          {r.filled_from_median.length > 0 && (
            <div className="panel" style={{ marginTop: "1rem" }}>
              <h3>{r.filled_from_median.length} yếu tố điền bằng trung vị tập train</h3>
              <DataTable
                rows={r.filled_from_median}
                columns={[
                  { key: "feature", label: "Yếu tố" },
                  { key: "mo_ta", label: "Ý nghĩa" },
                ]}
              />
            </div>
          )}
        </div>
      </div>
      <div className="panel">
        <div className="card-title-row">
          <h3>Vì sao hồ sơ này được chấm như vậy</h3>
          {r.shap && <Caveat>{r.shap.note}</Caveat>}
        </div>
        {r.shap ? (
          <>
            <DataTable
              rows={r.shap.factors}
              columns={[
                { key: "feature", label: "Yếu tố" },
                { key: "value", label: "Giá trị" },
                { key: "direction", label: "Đẩy rủi ro" },
                { key: "shap", label: "SHAP (log-odds)" },
                { key: "mo_ta", label: "Ý nghĩa" },
              ]}
            />
            <p className="muted">
              Điểm CatBoost {pct(r.shap.probability, 1)} (mức nền {pct(r.shap.base_probability, 1)}) - điểm
              xếp hạng chưa hiệu chỉnh, không phải PD ở trên.
            </p>
          </>
        ) : (
          <Card kind="warn">{r.shap_error}</Card>
        )}
      </div>
    </div>
  );
}
