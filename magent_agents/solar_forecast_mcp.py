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

KOREAN_AGENT_NAME = "태양광 발전량 예측 에이전트"
AGENT_KEY = "solar_forecast"

logging.basicConfig(
    level=logging.DEBUG,
    format=f"[%(asctime)s][%(levelname)s][solar_forecast_agent|{KOREAN_AGENT_NAME}] %(message)s",
)
logger = logging.getLogger(__name__)

_selected = (os.getenv("MAGENT_LOG_AGENT") or "").strip().lower()
if _selected and _selected not in {"all", "*", AGENT_KEY}:
    logging.disable(logging.CRITICAL)

mcp = FastMCP(name=f"solar_forecast_agent|{KOREAN_AGENT_NAME}")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = PROJECT_ROOT / ".magent_tmp"
MAX_TRAIN_TRIES = 4
MIN_TRIES_BEFORE_EARLY_STOP = 2
SOLAR_ACCEPTABLE_RMSE = float(os.getenv("MAGENT_SOLAR_ACCEPTABLE_RMSE", "20.0"))
TRAIN_CANDIDATES = [
    {
        "MAGENT_LEARNING_RATE": "0.05",
        "MAGENT_CAT_DEPTH": "4",
        "MAGENT_LGBM_N_ESTIMATORS": "150",
        "MAGENT_XGB_N_ESTIMATORS": "200",
        "MAGENT_XGB_MAX_DEPTH": "4",
    },
    {
        "MAGENT_LEARNING_RATE": "0.1",
        "MAGENT_CAT_DEPTH": "6",
        "MAGENT_LGBM_N_ESTIMATORS": "200",
        "MAGENT_XGB_N_ESTIMATORS": "300",
        "MAGENT_XGB_MAX_DEPTH": "6",
    },
    {
        "MAGENT_LEARNING_RATE": "0.2",
        "MAGENT_CAT_DEPTH": "8",
        "MAGENT_LGBM_N_ESTIMATORS": "250",
        "MAGENT_XGB_N_ESTIMATORS": "400",
        "MAGENT_XGB_MAX_DEPTH": "8",
    },
    {
        "MAGENT_LEARNING_RATE": "0.08",
        "MAGENT_CAT_DEPTH": "5",
        "MAGENT_LGBM_N_ESTIMATORS": "300",
        "MAGENT_XGB_N_ESTIMATORS": "500",
        "MAGENT_XGB_MAX_DEPTH": "5",
    },
]


def _python() -> str:
    venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    if venv_python.exists() and os.access(str(venv_python), os.X_OK):
        return str(venv_python)
    return sys.executable or "python3"


def _run_gpu_script(
    script_relpath: str,
    args: list[str] | None = None,
    timeout_seconds: float = 180.0,
    extra_env: dict[str, str] | None = None,
) -> dict:
    start = time.monotonic()
    script_path = str((PROJECT_ROOT / script_relpath).resolve())
    cmd = [_python(), script_path]
    if args:
        cmd.extend(args)

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        cmd,
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
        report_path = PROJECT_ROOT / f"solar_training_report_{ts}.log"
        lines: list[str] = []
        lines.append(f"timestamp={ts}")
        lines.append("agent=solar_forecast_agent")
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
        result_json = TMP_DIR / f"solar_train_{uuid4().hex}.json"
        env_for_train = dict(params)
        env_for_train["MAGENT_TRAIN_RESULT_PATH"] = str(result_json)
        run_info = _run_gpu_script(
            "solar_train/train.py",
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

        if run_info["returncode"] == 0 and payload and payload.get("direction") == "min":
            success_count += 1
            if best is None or float(payload["score"]) < float(best["payload"]["score"]):
                best = row
            if success_count >= MIN_TRIES_BEFORE_EARLY_STOP and float(best["payload"]["score"]) <= SOLAR_ACCEPTABLE_RMSE:
                break

    if best is None:
        report_file = _write_report(
            rounds=rounds,
            best_payload=None,
            error_message="solar training failed for all learning rates",
        )
        logger.error("solar training report written: %s", report_file)
        raise RuntimeError("solar training failed for all learning rates")

    best_payload = best["payload"]
    src_stack = Path(best_payload["stack_path"])
    src_scaler = Path(best_payload["scaler_path"])
    src_features = Path(best_payload["features_path"])

    dst_dir = PROJECT_ROOT / "gpu" / "solar_predict" / "model"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_stack = dst_dir / src_stack.name
    dst_scaler = dst_dir / src_scaler.name
    dst_features = dst_dir / src_features.name

    shutil.copy(src_stack, dst_stack)
    shutil.copy(src_scaler, dst_scaler)
    shutil.copy(src_features, dst_features)

    now = time.time()
    os.utime(dst_stack, (now, now))
    os.utime(dst_scaler, (now, now))
    os.utime(dst_features, (now, now))

    deployed = {
        "stack": str(dst_stack),
        "scaler": str(dst_scaler),
        "features": str(dst_features),
    }
    report_file = _write_report(rounds=rounds, best_payload=best_payload, deployed=deployed)
    logger.info("solar training report written: %s", report_file)

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
        "deployed_stack": str(dst_stack),
        "deployed_scaler": str(dst_scaler),
        "deployed_features": str(dst_features),
        "report_file": report_file,
    }


@mcp.tool()
def solar_forecast(location: str = "default") -> dict:
    logger.info("tool called: solar_forecast(location=%s)", location)
    start = time.monotonic()
    try:
        train_summary = _run_training_and_deploy()
        # gpu/solar_predict/solar_predict.py expects an input_date argument via __main__ block,
        # but currently it hardcodes a test date. We still run it and collect produced artifacts.
        run_info = _run_gpu_script("gpu/solar_predict/solar_predict.py")
        artifacts: list[str] = []
        result_dir = PROJECT_ROOT / "gpu" / "solar_predict" / "solar_predict_result"
        if result_dir.exists():
            artifacts.extend([str(p) for p in sorted(result_dir.glob("*.png"))[:20]])

        status = "completed" if run_info["returncode"] == 0 else "failed"
        result = {
            "agent": "solar_forecast_agent",
            "agent_ko": KOREAN_AGENT_NAME,
            "location": location,
            "status": status,
            "duration_seconds": round(time.monotonic() - start, 3),
            "gpu_script": "gpu/solar_predict/solar_predict.py",
            "training": train_summary,
            "artifacts": artifacts,
            **run_info,
        }
        logger.info("tool completed: %s", {"status": status, "artifact_count": len(artifacts)})
        return result
    except subprocess.TimeoutExpired as e:
        return {
            "agent": "solar_forecast_agent",
            "agent_ko": KOREAN_AGENT_NAME,
            "location": location,
            "status": "timeout",
            "duration_seconds": round(time.monotonic() - start, 3),
            "gpu_script": "gpu/solar_predict/solar_predict.py",
            "error": str(e),
        }
    except Exception as e:
        logger.exception("tool failed")
        return {
            "agent": "solar_forecast_agent",
            "agent_ko": KOREAN_AGENT_NAME,
            "location": location,
            "status": "failed",
            "duration_seconds": round(time.monotonic() - start, 3),
            "gpu_script": "gpu/solar_predict/solar_predict.py",
            "error": str(e),
        }


if __name__ == "__main__":
    logger.info("starting MCP stdio server")
    mcp.run(transport="stdio")
