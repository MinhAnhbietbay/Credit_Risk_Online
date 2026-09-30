import { useEffect, useRef, useState, type ReactNode } from "react";

export function InfoTooltip({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    function onClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, [open]);

  return (
    <span className="info-tooltip" ref={ref} onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button
        type="button"
        className="info-tooltip-trigger"
        aria-label="Xem chi tiết"
        onClick={() => setOpen((v) => !v)}
      >
      </button>
      {open && <div className="info-tooltip-popover">{children}</div>}
    </span>
  );
}
