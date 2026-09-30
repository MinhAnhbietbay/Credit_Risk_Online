from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.api.services.theme import COLORS, PLOT_LAYOUT, RISK_COLORS, risk_level

__all__ = ["risk_gauge", "confusion", "distribution", "band_chart", "calibration_chart",
           "bucket_chart", "stress_band_chart"]

def risk_gauge(pd_value: float, threshold: float) -> go.Figure:
    """Đồng hồ PD, có vạch đỏ đúng tại ngưỡng chặn đang áp dụng."""
    level, color = risk_level(pd_value, threshold)
    top = max(60, pd_value * 120)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pd_value * 100,
        number={"suffix": " %", "font": {"size": 34}},
        title={"text": f"Xác suất vỡ nợ — nhóm {level}", "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, top], "ticksuffix": "%"},
            "bar": {"color": color, "thickness": 0.72},
            "bgcolor": COLORS["surface"],
            "borderwidth": 1, "bordercolor": COLORS["grid"],
            "steps": [
                {"range": [0, threshold * 50], "color": "#eaf6f0"},
                {"range": [threshold * 50, threshold * 100], "color": "#fdf6e7"},
                {"range": [threshold * 100, top], "color": "#fdeeee"},
            ],
            "threshold": {"line": {"color": COLORS["danger"], "width": 4},
                          "thickness": 0.9, "value": threshold * 100},
        },
    ))
    fig.update_layout(**PLOT_LAYOUT, height=290)
    return fig

def confusion(df: pd.DataFrame) -> go.Figure:
    labeled = df.dropna(subset=["thuc_te"])
    cm = np.zeros((2, 2), dtype=int)
    for t in (0, 1):
        for p in (0, 1):
            cm[t, p] = int(((labeled["thuc_te"] == t) & (labeled["du_doan"] == p)).sum())
    fig = go.Figure(go.Heatmap(
        z=cm.tolist(), text=cm.tolist(), texttemplate="%{text}",
        x=["Dự đoán: trả được", "Dự đoán: vỡ nợ"],
        y=["Thực tế: trả được", "Thực tế: vỡ nợ"],
        colorscale=[[0, "#ffffff"], [1, COLORS["primary"]]], showscale=False))
    fig.update_layout(**PLOT_LAYOUT, height=340,
                      title="Ma trận nhầm lẫn trên phần đã có kết quả thật")
    fig.update_yaxes(autorange="reversed")
    return fig

def distribution(df: pd.DataFrame) -> go.Figure:
    labeled = df.dropna(subset=["thuc_te"])
    fig = go.Figure()
    for value, name, color in ((0, "Trả được nợ", COLORS["success"]),
                               (1, "Vỡ nợ", COLORS["danger"])):
        sub = labeled[labeled["thuc_te"] == value]
        if not sub.empty:
            fig.add_histogram(x=sub["pd"].tolist(), name=name, marker_color=color,
                              opacity=0.65, nbinsx=40)
    if not labeled.empty:
        fig.add_vline(x=float(labeled["nguong"].iloc[0]), line_dash="dash",
                      line_color=COLORS["text"], annotation_text="ngưỡng chặn")
    fig.update_layout(**PLOT_LAYOUT, height=340, barmode="overlay",
                      title="Phân bố PD theo kết quả thật", xaxis_title="PD",
                      yaxis_title="Số hồ sơ")
    return fig

def band_chart(frame: pd.DataFrame) -> go.Figure:
    data = frame.assign(nhom=frame["nhom"].astype(str))
    fig = px.bar(data, x="nhom", y="so_ho_so", text="so_ho_so", color="nhom",
                 color_discrete_sequence=list(RISK_COLORS.values()))
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(**PLOT_LAYOUT, height=340, showlegend=False,
                      title="Số hồ sơ theo nhóm rủi ro", xaxis_title="nhóm PD",
                      yaxis_title="số hồ sơ")
    return fig

def calibration_chart(frame: pd.DataFrame) -> go.Figure:
    have = frame.dropna(subset=["ti_le_vo_no_that"])
    x = have["nhom"].astype(str).tolist()
    fig = go.Figure()
    fig.add_bar(x=x, y=have["pd_tb"].tolist(), name="PD model dự đoán",
                marker_color=COLORS["primary"])
    fig.add_bar(x=x, y=have["ti_le_vo_no_that"].tolist(), name="Tỉ lệ vỡ nợ thật",
                marker_color=COLORS["accent"])
    fig.update_layout(**PLOT_LAYOUT, height=340, barmode="group",
                      title="Model dự đoán so với thực tế, theo nhóm",
                      yaxis_tickformat=".1%", yaxis_title="tỉ lệ")
    return fig

def bucket_chart(feature: str, frame: pd.DataFrame) -> go.Figure:
    fig = px.bar(frame, x="nhom", y="ti_le_vo_no", text="so_ho_so", color="ti_le_vo_no",
                 color_continuous_scale=["#1baf7a", "#eda100", "#c2383f"])
    fig.update_traces(
        texttemplate="n=%{text:,}", textposition="outside", cliponaxis=False,
        customdata=frame[["so_ho_so"]].to_numpy(),
        hovertemplate=("Nhóm (khoảng giá trị): %{x}<br>Tỉ lệ vỡ nợ: %{y:.2%}"
                       "<br>Số hồ sơ: %{customdata[0]:,}<extra></extra>"))
    fig.update_layout(**PLOT_LAYOUT, height=380, coloraxis_showscale=False,
                      title=f"Tỉ lệ vỡ nợ theo phân vị của {feature}",
                      xaxis_title="phân vị (thấp → cao)", yaxis_title="tỉ lệ vỡ nợ",
                      yaxis_tickformat=".1%")
    # chừa chỗ phía trên để nhãn n=... của cột cao nhất không bị cắt
    fig.update_yaxes(range=[0, float(frame["ti_le_vo_no"].max()) * 1.2])
    fig.update_xaxes(showticklabels=False)
    return fig

STRESS_BANDS = {"band_0_0.05": "PD < 5%", "band_0.05_0.15": "5–15%",
                "band_0.15_0.3": "15–30%", "band_0.3_plus": "≥ 30%"}

def stress_band_chart(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for (col, label), color in zip(STRESS_BANDS.items(), RISK_COLORS.values()):
        if col in table.columns:
            fig.add_bar(x=table["scenario"].tolist(), y=(table[col] * 100).tolist(),
                        name=label, marker_color=color)
    fig.update_layout(**PLOT_LAYOUT, height=400, barmode="stack",
                      title="Dịch chuyển nhóm rủi ro theo kịch bản", yaxis_title="% hồ sơ trong mẫu")
    return fig