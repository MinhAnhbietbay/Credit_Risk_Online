from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
HOME_CREDIT_ZIP = DATA_DIR / "home-credit-default-risk.zip"  
HOME_CREDIT_DIR = DATA_DIR / "home-credit-default-risk"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
OOF_DIR = OUTPUT_DIR / "oof"
EVIDENCE_DIR = OUTPUT_DIR / "evidence"

for d in [PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR, OOF_DIR, EVIDENCE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5