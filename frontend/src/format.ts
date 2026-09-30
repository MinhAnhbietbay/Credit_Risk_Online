export const pct = (v: number | null | undefined, digits = 2): string =>
  v == null ? "-" : `${(v * 100).toFixed(digits)}%`;

export const signedPct = (v: number, digits = 2): string =>
  `${v >= 0 ? "+" : ""}${(v * 100).toFixed(digits)}%`;

export const int = (v: number | null | undefined): string =>
  v == null ? "-" : v.toLocaleString("en-US");
