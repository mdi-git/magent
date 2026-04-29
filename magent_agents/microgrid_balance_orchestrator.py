import asyncio
import os
import sys
import time
import logging
from pathlib import Path
from typing import Any

from agents import Agent, Runner, enable_verbose_stdout_logging
from agents.mcp import MCPServerManager, MCPServerStdio


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORCHESTRATOR_KOREAN_NAME = "마이크로그리드 수급 밸런스 최적화 오케스트레이터"

AGENT_KOREAN_NAMES: dict[str, str] = {
    "solar_forecast": "태양광 발전량 예측 에이전트",
    "wind_forecast": "풍력 발전량 예측 에이전트",
    "wind_anomaly": "풍력 이상 탐지 에이전트",
    "solar_anomaly": "태양광 이상 탐지 에이전트",
    "consumption_forecast": "전력 소비 예측 에이전트",
}

AGENT_ORDER: list[str] = [
    "solar_forecast",
    "wind_forecast",
    "wind_anomaly",
    "solar_anomaly",
    "consumption_forecast",
]

logging.basicConfig(
    level=logging.DEBUG,
    format=f"[%(asctime)s][%(levelname)s][orchestrator|{ORCHESTRATOR_KOREAN_NAME}] %(message)s",
)
logger = logging.getLogger(__name__)


def _python() -> str:
    venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    if venv_python.exists() and os.access(str(venv_python), os.X_OK):
        return str(venv_python)
    return sys.executable or "python3"


def _normalize_log_agent(log_agent: str | None) -> str | None:
    if not log_agent:
        return None

    v = log_agent.strip().lower()
    if v in {"all", "*"}:
        return "all"
    if v in {"orch", "orchestrator", "microgrid_balance_optimization_orchestrator"}:
        return "orchestrator"

    if v.isdigit():
        idx = int(v)
        if 1 <= idx <= len(AGENT_ORDER):
            return AGENT_ORDER[idx - 1]
        return None

    if v.endswith("_agent"):
        v = v[: -len("_agent")]

    if v in AGENT_KOREAN_NAMES:
        return v

    for key, ko in AGENT_KOREAN_NAMES.items():
        if v == ko.strip().lower():
            return key
    return None


def _mcp_server(name: str, korean_name: str, script_relpath: str) -> MCPServerStdio:
    script_path = str((PROJECT_ROOT / script_relpath).resolve())
    env = dict(os.environ)
    env.setdefault("FASTMCP_LOG_LEVEL", "DEBUG")
    env.setdefault("FASTMCP_DEBUG", "true")
    return MCPServerStdio(
        name=f"{name}|{korean_name}",
        params={
            "command": _python(),
            "args": [script_path],
            "cwd": str(PROJECT_ROOT),
            "env": env,
        },
        client_session_timeout_seconds=3600,
        cache_tools_list=True,
    )


async def run_orchestrator(log_agent: str | None = None) -> dict[str, Any]:
    normalized = _normalize_log_agent(log_agent)
    if not normalized or normalized in {"all", "orchestrator"}:
        enable_verbose_stdout_logging()
    if normalized and normalized != "all":
        os.environ["MAGENT_LOG_AGENT"] = normalized
    elif normalized == "all":
        os.environ.pop("MAGENT_LOG_AGENT", None)
    else:
        os.environ.pop("MAGENT_LOG_AGENT", None)

    if normalized and normalized not in {"all", "orchestrator"}:
        logger.disabled = True
        logging.disable(logging.CRITICAL)
    else:
        logger.disabled = False
        logging.disable(logging.NOTSET)

    logger.info("starting orchestration")

    servers = [
        _mcp_server("solar_forecast", AGENT_KOREAN_NAMES["solar_forecast"], "magent_agents/solar_forecast_mcp.py"),
        _mcp_server("wind_forecast", AGENT_KOREAN_NAMES["wind_forecast"], "magent_agents/wind_forecast_mcp.py"),
        _mcp_server("wind_anomaly", AGENT_KOREAN_NAMES["wind_anomaly"], "magent_agents/wind_anomaly_mcp.py"),
        _mcp_server("solar_anomaly", AGENT_KOREAN_NAMES["solar_anomaly"], "magent_agents/solar_anomaly_mcp.py"),
        _mcp_server(
            "consumption_forecast",
            AGENT_KOREAN_NAMES["consumption_forecast"],
            "magent_agents/consumption_forecast_mcp.py",
        ),
    ]

    start = time.monotonic()
    try:
        async with MCPServerManager(servers, connect_in_parallel=True, strict=True) as manager:
            logger.info(
                "connected MCP servers: %s",
                [
                    {"server": s.name, "korean": s.name.split("|", 1)[1] if "|" in s.name else ""}
                    for s in manager.active_servers
                ],
            )

            orchestrator = Agent(
                name="microgrid_balance_optimization_orchestrator",
                instructions=(
                    "You are the Microgrid generation/consumption balance optimization orchestrator. "
                    "You MUST call each available MCP tool exactly once to collect outputs, then return a concise JSON summary. "
                    "Do not call any tool more than once. Output only JSON."
                ),
                mcp_servers=manager.active_servers,
            )

            prompt = (
                "Run orchestration now. Use these parameters when calling tools:\n"
                "- solar_forecast(location='seoul')\n"
                "- wind_forecast(location='jeju')\n"
                "- wind_anomaly_detect(asset_id='wind_turbine_01')\n"
                "- solar_anomaly_detect(asset_id='solar_array_01')\n"
                "- consumption_forecast(customer_group='residential')\n"
                "After tool calls, return JSON with keys: agent, status, duration_seconds, results."
            )

            logger.info("calling Runner.run()")
            run_result = await Runner.run(orchestrator, prompt, max_turns=5)
            logger.info("Runner.run() completed")

            return {
                "agent": "microgrid_balance_optimization_orchestrator",
                "agent_ko": ORCHESTRATOR_KOREAN_NAME,
                "status": "completed",
                "duration_seconds": round(time.monotonic() - start, 3),
                "final_output": run_result.final_output,
            }
    except Exception:
        logger.exception("orchestration failed")
        raise


if __name__ == "__main__":
    out = asyncio.run(run_orchestrator())
    print(out)
