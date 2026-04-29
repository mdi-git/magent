import asyncio
import argparse

from magent_agents.microgrid_balance_orchestrator import run_orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "log_agent",
        nargs="?",
        default=None,
        help="Log target agent (name or number). Examples: 1, solar_forecast, wind_anomaly, orchestrator, all",
    )
    parser.add_argument(
        "--log-agent",
        dest="log_agent_flag",
        default=None,
        help="Same as positional log_agent",
    )
    args = parser.parse_args()

    log_agent = args.log_agent_flag or args.log_agent
    result = asyncio.run(run_orchestrator(log_agent=log_agent))
    print(result)


if __name__ == "__main__":
    main()
