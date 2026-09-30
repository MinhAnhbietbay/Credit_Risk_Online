import { type FormEvent, useState } from "react";

import type { FormField } from "../api/predict";

type Props = {
  fields: FormField[];
  nTotal: number;
  busy: boolean;
  onSubmit: (values: Record<string, number>) => void;
};

// Nhóm hiển thị theo tên feature. Feature chưa có trong bảng rơi vào "Khác",
// nên top SHAP đổi thì form vẫn hiển thị đủ.
const GROUPS: { title: string; features: string[] }[] = [
  {
    title: "Điểm tín dụng nguồn ngoài",
    features: ["EXT_SOURCE_MEAN", "EXT_SOURCE_1", "EXT_SOURCE_3", "EXT_SOURCE_MAX", "EXT_SOURCE_MIN"],
  },
  { title: "Khoản vay", features: ["AMT_ANNUITY", "CREDIT_TERM"] },
  {
    title: "Nhân thân và việc làm",
    features: ["DAYS_BIRTH", "DAYS_EMPLOYED", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE", "OWN_CAR_AGE"],
  },
  {
    title: "Lịch sử trả nợ",
    features: ["INST_LATE_RATIO", "PREV_DAYS_LAST_DUE_MAX", "PREV_DAYS_LAST_DUE_1ST_VERSION_MAX"],
  },
];

function groupFields(fields: FormField[]): { title: string; fields: FormField[] }[] {
  const known = new Set(GROUPS.flatMap((g) => g.features));
  const groups = GROUPS.map((g) => ({
    title: g.title,
    fields: g.features
      .map((name) => fields.find((f) => f.feature === name))
      .filter((f): f is FormField => f !== undefined),
  }));
  groups.push({ title: "Khác", fields: fields.filter((f) => !known.has(f.feature)) });
  return groups.filter((g) => g.fields.length > 0);
}

// Làm tròn để hiển thị: số nguyên giữ nguyên, số thực lấy 4 chữ số có nghĩa.
function round(v: number): number {
  return Number.isInteger(v) ? v : Number(v.toPrecision(4));
}

function fmt(v: number): string {
  return round(v).toLocaleString("en-US", { maximumFractionDigits: 6 });
}


export function ApplicantForm({ fields, nTotal, busy, onSubmit }: Props) {
  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries(fields.map((f) => [f.feature, ""])),
  );
  const set = (name: string, v: string) => setValues((old) => ({ ...old, [name]: v }));

  function submit(e: FormEvent) {
    e.preventDefault();
    // ô để trống thì không gửi: backend tự điền trung vị cho trường đó
    onSubmit(
      Object.fromEntries(
        Object.entries(values)
          .filter(([, v]) => v.trim() !== "")
          .map(([k, v]) => [k, Number(v)]),
      ),
    );
  }

  return (
    <form onSubmit={submit}>
      <p className="muted">Số mờ trong ô chỉ là gợi ý. Ô số để trống dùng trung vị của tập train; ô chọn để "Chưa rõ" dùng nhóm phổ biến nhất của tập train.</p>
      <div className="form-groups">
      {groupFields(fields).map((g) => (
        <fieldset key={g.title} className="form-group">
          <legend>{g.title}</legend>
          {g.fields.map((f) => (
            <label key={f.feature} className="form-row">
              <span className="form-label">
                <span className="field-name">{f.feature}</span>
                <span className="muted">{f.mo_ta}</span>
                {!f.categorical && (
                  <span className="range-hint">
                    Khoảng cho phép: {fmt(f.min)} đến {fmt(f.max)}
                  </span>
                )}
              </span>
              <span className="form-control">
                {f.categorical ? (
                  <select value={values[f.feature]} onChange={(e) => set(f.feature, e.target.value)}>
                    <option value="">Chưa rõ</option>
                    {f.options.map((o) => (
                      <option key={o.code} value={o.code}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    min={f.min}
                    max={f.max}
                    step="any"
                    placeholder={`VD: ${fmt(f.median)}`}
                    value={values[f.feature]}
                    onChange={(e) => set(f.feature, e.target.value)}
                  />
                )}
              </span>
            </label>
          ))}
        </fieldset>
      ))}
      </div>
      <p className="muted">
        Form hỏi {fields.length}/{nTotal} yếu tố quan trọng nhất theo SHAP. {nTotal - fields.length}{" "}
        yếu tố còn lại, cùng mọi ô bạn để trống, được điền <b>trung vị của tập train</b> - PD tính ra
        là ước lượng cho một hồ sơ "trung bình" ở những mặt không được hỏi.
      </p>
      <div className="action-end">
        <button type="submit" disabled={busy}>
          {busy ? "Đang chấm…" : "Chấm điểm"}
        </button>
      </div>
    </form>
  );
}
