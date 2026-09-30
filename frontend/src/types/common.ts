export type Figure = { data: object[]; layout: Record<string, unknown> };

export type Cell = string | number | boolean | null;
export type Row = Record<string, Cell>;

export type PageProps = { onChanged: () => void };
