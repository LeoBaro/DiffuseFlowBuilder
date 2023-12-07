from pathlib import Path

ROOT = Path(__file__).parent.absolute().parent.parent

EVALUATION_RUNS_LOG_DIR  = ROOT / "ml_logs" / "evaluation_runs"
EVALUATION_RUNS_LOG_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_RUNS_LOG_DIR  = ROOT / "ml_logs" / "training_runs"
TRAINING_RUNS_LOG_DIR.mkdir(parents=True, exist_ok=True)

PATH_LOGS = ROOT / "logs"