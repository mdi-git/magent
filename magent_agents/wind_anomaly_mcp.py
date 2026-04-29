import time
import logging
import os
import subprocess
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

KOREAN_AGENT_NAME = "풍력 이상 탐지 에이전트"
AGENT_KEY = "wind_anomaly"

logging.basicConfig(
    level=logging.DEBUG,
    format=f"[%(asctime)s][%(levelname)s][wind_anomaly_detection_agent|{KOREAN_AGENT_NAME}] %(message)s",
)
logger = logging.getLogger(__name__)

_selected = (os.getenv("MAGENT_LOG_AGENT") or "").strip().lower()
if _selected and _selected not in {"all", "*", AGENT_KEY}:
    logging.disable(logging.CRITICAL)

mcp = FastMCP(name=f"wind_anomaly_detection_agent|{KOREAN_AGENT_NAME}")


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _python() -> str:
    return sys.executable or "python3"


def _run_gpu_script(script_relpath: str, timeout_seconds: float = 300.0) -> dict:
    start = time.monotonic()
    script_path = str((PROJECT_ROOT / script_relpath).resolve())

    proc = subprocess.run(
        [_python(), script_path],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=dict(os.environ),
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


@mcp.tool()
def wind_anomaly_detect(asset_id: str = "wind_turbine_01") -> dict:
    logger.info("tool called: wind_anomaly_detect(asset_id=%s)", asset_id)
    start = time.monotonic()
    try:
        run_info = _run_gpu_script("gpu/wind_anomaly/wind_anomaly.py")
        artifacts: list[str] = []
        result_dir = PROJECT_ROOT / "gpu" / "wind_anomaly" / "wind_anomaly_results"
        if result_dir.exists():
            artifacts.extend([str(p) for p in sorted(result_dir.glob("*.png"))[:20]])

        status = "completed" if run_info["returncode"] == 0 else "failed"
        result = {
            "agent": "wind_anomaly_detection_agent",
            "agent_ko": KOREAN_AGENT_NAME,
            "asset_id": asset_id,
            "status": status,
            "duration_seconds": round(time.monotonic() - start, 3),
            "gpu_script": "gpu/wind_anomaly/wind_anomaly.py",
            "artifacts": artifacts,
            **run_info,
        }
        logger.info("tool completed: %s", {"status": status, "artifact_count": len(artifacts)})
        return result
    except subprocess.TimeoutExpired as e:
        return {
            "agent": "wind_anomaly_detection_agent",
            "agent_ko": KOREAN_AGENT_NAME,
            "asset_id": asset_id,
            "status": "timeout",
            "duration_seconds": round(time.monotonic() - start, 3),
            "gpu_script": "gpu/wind_anomaly/wind_anomaly.py",
            "error": str(e),
        }
    except Exception as e:
        logger.exception("tool failed")
        return {
            "agent": "wind_anomaly_detection_agent",
            "agent_ko": KOREAN_AGENT_NAME,
            "asset_id": asset_id,
            "status": "failed",
            "duration_seconds": round(time.monotonic() - start, 3),
            "gpu_script": "gpu/wind_anomaly/wind_anomaly.py",
            "error": str(e),
        }


if __name__ == "__main__":
    logger.info("starting MCP stdio server")
    mcp.run(transport="stdio")
