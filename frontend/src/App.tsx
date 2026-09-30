import { type ComponentType, useCallback, useState } from "react";

import { Sidebar } from "./components/layout/Sidebar";
import BusinessInsightsPage from "./pages/BusinessInsightsPage";
import PredictionsPage from "./pages/PredictionsPage";
import PredictPage from "./pages/PredictPage";
import RetrainPage from "./pages/RetrainPage";
import RiskSegmentationPage from "./pages/RiskSegmentationPage";
import StressTestPage from "./pages/StressTestPage";
import type { PageProps } from "./types/common";

// Thứ tự và tên tab giữ nguyên như bản Streamlit.
const TABS: { title: string; Page: ComponentType<PageProps> }[] = [
  { title: "Dự đoán", Page: PredictPage },
  { title: "Train lại", Page: RetrainPage },
  { title: "Dự đoán vs thực tế", Page: PredictionsPage },
  { title: "Insight nghiệp vụ", Page: BusinessInsightsPage },
  { title: "Phân khúc rủi ro", Page: RiskSegmentationPage },
  { title: "Stress test", Page: StressTestPage },
];

export default function App() {
  const [active, setActive] = useState(0);
  const [refreshKey, setRefreshKey] = useState(0);
  const onChanged = useCallback(() => setRefreshKey((k) => k + 1), []);
  const { Page } = TABS[active];

  return (
    <div className="app-shell">
      <header className="topbar">
        <h1>Rủi ro tín dụng</h1>
        <nav className="tabs">
          {TABS.map((t, i) => (
            <button key={t.title} className={i === active ? "tab active" : "tab"} onClick={() => setActive(i)}>
              {t.title}
            </button>
          ))}
        </nav>
        <span aria-hidden="true" />
      </header>
      <div className="layout">
        <Sidebar refreshKey={refreshKey} />
        <main>
          <Page onChanged={onChanged} />
        </main>
      </div>
    </div>
  );
}
