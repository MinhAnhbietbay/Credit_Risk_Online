import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

import type { Figure } from "../../types/common";

const Plot = createPlotlyComponent(Plotly);

export function PlotlyChart({ figure }: { figure: Figure }) {
  return (
    <Plot
      data={figure.data}
      layout={{ ...figure.layout, autosize: true }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
