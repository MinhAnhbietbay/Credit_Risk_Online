import { pct } from "../../format";

type Props = { pd: number; level: string; color: string; decision: string; threshold: number };

export function Badge({ pd, level, color, decision, threshold }: Props) {
  return (
    <>
      <div className="badge" style={{ background: color }}>
        PD {pct(pd)} · nhóm {level} · {decision}
      </div>
      <p className="muted">
        Ngưỡng chặn đang áp dụng: {threshold.toFixed(4)} (chặn 21.9% hồ sơ điểm cao nhất - theo tỉ
        lệ từ chối lịch sử của Home Credit
      </p>
    </>
  );
}
