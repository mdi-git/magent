import time
import logging
import os
import subprocess
import sys
import json
import shutil
from datetime import datetime
from uuid import uuid4
from pathlib import Path

from mcp.server.fastmcp import FastMCP

KOREAN_AGENT_NAME = "풍력 발전량 예측 에이전트"
AGENT_KEY = "wind_forecast"

logging.basicConfig(
    level=logging.DEBUG,
    format=f"[%(asctime)s][%(levelname)s][wind_forecast_agent|{KOREAN_AGENT_NAME}] %(message)s",
)
logger = logging.getLogger(__name__)

_selected = (os.getenv("MAGENT_LOG_AGENT") or "").strip().lower()
if _selected and _selected not in {"all", "*", AGENT_KEY}:
    logging.disable(logging.CRITICAL)

mcp = FastMCP(name=f"wind_forecast_agent|{KOREAN_AGENT_NAME}")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = PROJECT_ROOT / ".magent_tmp"
MAX_TRAIN_TRIES = 4
MIN_TRIES_BEFORE_EARLY_STOP = 2
WIND_ACCEPTABLE_R2 = float(os.getenv("MAGENT_WIND_ACCEPTABLE_R2", "0.85"))
TRAIN_CANDIDATES = [
    {
        "MAGENT_LEARNING_RATE": "0.02",
        "MAGENT_ITERATIONS": "60000",
        "MAGENT_DEPTH": "3",
        "MAGENT_L2_LEAF_REG": "3",
    },
    {
        "MAGENT_LEARNING_RATE": "0.04",
        "MAGENT_ITERATIONS": "100000",
        "MAGENT_DEPTH": "2",
        "MAGENT_L2_LEAF_REG": "3",
    },
    {
        "MAGENT_LEARNING_RATE": "0.08",
        "MAGENT_ITERATIONS": "80000",
        "MAGENT_DEPTH": "4",
        "MAGENT_L2_LEAF_REG": "5",
    },
    {
        "MAGENT_LEARNING_RATE": "0.03",
        "MAGENT_ITERATIONS": "120000",
        "MAGENT_DEPTH": "5",
        "MAGENT_L2_LEAF_REG": "7",
    },
]


def _python() -> str:
    venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    if venv_python.exists() and os.access(str(venv_python), os.X_OK):
        return str(venv_python)
    return sys.executable or "python3"


def _run_gpu_script(
    script_relpath: str,
    timeout_seconds: float = 120.0,
    extra_env: dict[str, str] | None = None,
) -> dict:
    start = time.monotonic()
    script_path = str((PROJECT_ROOT / script_relpath).resolve())

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        [_python(), script_path],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=env,
    )
    duration = round(time.monotonic() - start, 3)
    stdout = (proc.stdout or "")
    stderr = (proc.stderr or "")

    return {
        "returncode": proc.returncode,
        "duration_seconds": duration,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def _run_training_and_deploy() -> dict:
    def _write_report(
        rounds: list[dict],
        best_payload: dict | None,
        deployed: dict | None = None,
        error_message: str | None = None,
    ) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        report_path = PROJECT_ROOT / f"wind_training_report_{ts}.log"
        lines: list[str] = []
        lines.append(f"timestamp={ts}")
        lines.append("agent=wind_forecast_agent")
        lines.append(f"status={'failed' if error_message else 'completed'}")
        if error_message:
            lines.append(f"error={error_message}")
        lines.append("")
        lines.append("[rounds]")
        for idx, r in enumerate(rounds, start=1):
            payload = r.get("payload") or {}
            run = r.get("run") or {}
            lines.append(
                f"#{idx} lr={r.get('learning_rate')} returncode={run.get('returncode')} "
                f"metric={payload.get('metric_name')} score={payload.get('score')} "
                f"params={json.dumps(r.get('params') or {}, ensure_ascii=False)}"
            )
        lines.append("")
        lines.append("[best]")
        if best_payload:
            lines.append(
                f"learning_rate={best_payload.get('learning_rate')} "
                f"metric={best_payload.get('metric_name')} score={best_payload.get('score')}"
            )
            lines.append(f"params={json.dumps(best_payload.get('params') or {}, ensure_ascii=False)}")
        else:
            lines.append("none")
        if deployed:
            lines.append("")
            lines.append("[deployed]")
            for k, v in deployed.items():
                lines.append(f"{k}={v}")

        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(report_path)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    rounds: list[dict] = []
    best: dict | None = None

    success_count = 0
    for params in TRAIN_CANDIDATES[:MAX_TRAIN_TRIES]:
        result_json = TMP_DIR / f"wind_train_{uuid4().hex}.json"
        env_for_train = dict(params)
        env_for_train["MAGENT_TRAIN_RESULT_PATH"] = str(result_json)
        run_info = _run_gpu_script(
            "wind_predict_train/wind_predict_train.py",
            timeout_seconds=1800.0,
            extra_env=env_for_train,
        )

        payload = None
        if result_json.exists():
            try:
                payload = json.loads(result_json.read_text(encoding="utf-8"))
            except Exception:
                payload = None

        row = {
            "learning_rate": float(params.get("MAGENT_LEARNING_RATE", "0")),
            "params": params,
            "run": run_info,
            "payload": payload,
        }
        rounds.append(row)

        if run_info["returncode"] == 0 and payload and payload.get("direction") == "max":
            success_count += 1
            if best is None or float(payload["score"]) > float(best["payload"]["score"]):
                best = row
            if success_count >= MIN_TRIES_BEFORE_EARLY_STOP and float(best["payload"]["score"]) >= WIND_ACCEPTABLE_R2:
                break

    if best is None:
        report_file = _write_report(
            rounds=rounds,
            best_payload=None,
            error_message="wind training failed for all learning rates",
        )
        logger.error("wind training report written: %s", report_file)
        raise RuntimeError("wind training failed for all learning rates")

    best_payload = best["payload"]
    best_model = Path(best_payload["model_path"])
    best_feature = Path(best_payload["feature_path"])

    dst_dir = PROJECT_ROOT / "gpu" / "wind_predict" / "cbm"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_model = dst_dir / "wind_model_1h.cbm"
    dst_feature = dst_dir / "wind_features_1h.joblib"

    shutil.copy(best_model, dst_model)
    shutil.copy(best_feature, dst_feature)

    deployed = {
        "model": str(dst_model),
        "feature": str(dst_feature),
    }
    report_file = _write_report(rounds=rounds, best_payload=best_payload, deployed=deployed)
    logger.info("wind training report written: %s", report_file)

    return {
        "best_learning_rate": best_payload["learning_rate"],
        "best_metric_name": best_payload["metric_name"],
        "best_score": best_payload["score"],
        "best_direction": best_payload["direction"],
        "rounds": [
            {
                "learning_rate": r["learning_rate"],
                "returncode": r["run"]["returncode"],
                "score": (r["payload"] or {}).get("score"),
                "metric_name": (r["payload"] or {}).get("metric_name"),
                "params": r.get("params"),
            }
            for r in rounds
        ],
        "deployed_model": str(dst_model),
        "deployed_feature": str(dst_feature),
        "report_file": report_file,
    }


@mcp.tool()
def wind_forecast(location: str = "default") -> dict:
    logger.info("tool called: wind_forecast(location=%s)", location)
    start = time.monotonic()
    try:
        train_summary = _run_training_and_deploy()
        run_info = _run_gpu_script("gpu/wind_predict/wind_predict.py")
        artifacts: list[str] = []
        expected_csv = PROJECT_ROOT / "gpu" / "wind_predict" / "wind_predict_1h_results" / "wind_prediction_1h_result.csv"
        if expected_csv.exists():
            artifacts.append(str(expected_csv))

        status = "completed" if run_info["returncode"] == 0 else "failed"
        result = {
            "agent": "wind_forecast_agent",
            "agent_ko": KOREAN_AGENT_NAME,
            "location": location,
            "status": status,
            "duration_seconds": round(time.monotonic() - start, 3),
            "gpu_script": "gpu/wind_predict/wind_predict.py",
            "training": train_summary,
            "artifacts": artifacts,
            **run_info,
        }
        logger.info("tool completed: %s", {"status": status, "artifacts": artifacts})
        return result
    except subprocess.TimeoutExpired as e:
        return {
            "agent": "wind_forecast_agent",
            "agent_ko": KOREAN_AGENT_NAME,
            "location": location,
            "status": "timeout",
            "duration_seconds": round(time.monotonic() - start, 3),
            "gpu_script": "gpu/wind_predict/wind_predict.py",
            "error": str(e),
        }
    except Exception as e:
        logger.exception("tool failed")
        return {
            "agent": "wind_forecast_agent",
            "agent_ko": KOREAN_AGENT_NAME,
            "location": location,
            "status": "failed",
            "duration_seconds": round(time.monotonic() - start, 3),
            "gpu_script": "gpu/wind_predict/wind_predict.py",
            "error": str(e),
        }


if __name__ == "__main__":
    logger.info("starting MCP stdio server")
    mcp.run(transport="stdio")
