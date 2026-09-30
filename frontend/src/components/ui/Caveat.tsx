import type { ReactNode } from "react";

import { InfoTooltip } from "./InfoTooltip";

export function Caveat({ children }: { children: ReactNode }) {
  return (
    <InfoTooltip>
      <p>{children}</p>
    </InfoTooltip>
  );
}
