from __future__ import annotations

__all__ = ["COLORS", "RISK_COLORS", "PLOT_LAYOUT", "risk_level", "decision_text"]

COLORS = {
    "primary": "#2a78d6",
    "accent": "#eb6834",
    "success": "#1baf7a",
    "warning": "#eda100",
    "danger": "#c2383f",
    "text": "#1c1b17",
    "muted": "#5c5b55",
    "grid": "#e6e5e1",
    "surface": "#ffffff",
    "bg": "rgba(0,0,0,0)",
}

RISK_COLORS = {"Thấp": COLORS["success"], "Trung bình": COLORS["warning"],
               "Cao": COLORS["accent"], "Rất cao": COLORS["danger"]}

PLOT_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(color=COLORS["text"], family="Inter, system-ui, sans-serif", size=13),
    margin=dict(t=48, b=36, l=48, r=24),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
)

def risk_level(pd_value: float, threshold: float) -> tuple[str, str]:
    """(tên nhóm, màu). Mốc neo vào ngưỡng chặn đã chốt ở Task 10, không phải số tròn tuỳ ý."""
    if pd_value >= threshold * 2.5:
        return "Rất cao", RISK_COLORS["Rất cao"]
    if pd_value >= threshold:
        return "Cao", RISK_COLORS["Cao"]
    if pd_value >= threshold * 0.5:
        return "Trung bình", RISK_COLORS["Trung bình"]
    return "Thấp", RISK_COLORS["Thấp"]

def decision_text(pd_value: float, threshold: float) -> str:
    return ("TỪ CHỐI / chuyển thẩm định thủ công" if pd_value >= threshold
            else "ĐỦ ĐIỀU KIỆN xét tiếp")
