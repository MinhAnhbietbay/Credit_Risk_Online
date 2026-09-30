import type { ReactNode } from "react";

export type CardKind = "info" | "ok" | "warn" | "danger";

export function Card({ kind = "info", children }: { kind?: CardKind; children: ReactNode }) {
  return <div className={`card ${kind}`}>{children}</div>;
}
