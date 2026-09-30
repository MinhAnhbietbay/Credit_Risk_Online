import { Card } from "./Card";

export function Loading() {
  return <p className="muted">Đang tải…</p>;
}

export function ErrorBox({ message }: { message: string }) {
  return <Card kind="danger">{message}</Card>;
}
