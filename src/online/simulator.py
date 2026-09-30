from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import RANDOM_SEED
from src.ph1_credit_risk.features.selection import FEATURES_TRAIN_PATH, PROTECTED
from src.online.schema import Applicant

__all__ = ["reveal_labels", "available_to_reveal"]


def available_to_reveal(session: Session) -> int:
    from sqlalchemy import func
    return int(session.execute(
        select(func.count()).select_from(Applicant)
        .where(Applicant.target.is_(None), Applicant.is_holdout.is_(False))
    ).scalar() or 0)

def reveal_labels(session: Session, n: int = 500, seed: int = RANDOM_SEED) -> dict:
    ids = session.execute(
        select(Applicant.sk_id_curr)
        .where(Applicant.target.is_(None), Applicant.is_holdout.is_(False))
        .order_by(Applicant.sk_id_curr).limit(int(n))
    ).scalars().all()
    if not ids:
        return {"n_revealed": 0, "n_remaining": 0, "n_default": 0}

    truth = pd.read_parquet(FEATURES_TRAIN_PATH, columns=list(PROTECTED))
    truth = truth[truth["SK_ID_CURR"].isin(ids)].set_index("SK_ID_CURR")["TARGET"]

    now, n_default = datetime.now(UTC), 0
    for sk in ids:
        if sk not in truth.index:
            continue
        a = session.get(Applicant, int(sk))
        a.target = int(truth.loc[sk]); a.label_revealed_at = now
        n_default += a.target
    session.commit()

    return {"n_revealed": len(ids), "n_remaining": available_to_reveal(session),
            "n_default": int(n_default)}
