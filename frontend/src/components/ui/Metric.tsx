export function Metric({ label, value, help }: { label: string; value: string; help?: string }) {
  return (
    <div className="metric" title={help}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}
