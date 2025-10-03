import argparse
import logging
import sys
import os
from typing import List

# Ensure project root on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
	sys.path.append(CURRENT_DIR)

from config import Config
from utils.fetch_data import DataFetcher


def parse_args() -> List[str]:
	parser = argparse.ArgumentParser(description="Start realtime streaming for tickers")
	parser.add_argument(
		"--tickers",
		type=str,
		help="Comma-separated list of tickers (e.g., VCB,VHM,VNM). Defaults to Config.TICKERS_WATCHLIST",
		default="",
	)
	args = parser.parse_args()
	if args.tickers:
		return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
	# Fallback to configured watchlist
	return list(getattr(Config, "TICKERS_WATCHLIST", []))


def main():
	logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
	logger = logging.getLogger("realtime_stream")

	tickers = parse_args()
	if not tickers:
		logger.error("No tickers provided and Config.TICKERS_WATCHLIST is empty.")
		sys.exit(1)

	logger.info(f"Starting realtime stream for {len(tickers)} tickers: {tickers}")

	fetcher = DataFetcher()
	if fetcher.client is None:
		logger.error("FiinQuantX client not initialized. Check credentials/network.")
		sys.exit(2)

	try:
		# This will run until interrupted and write snapshots to REALTIME_DATA_DIR
		fetcher.fetch_realtime_data(tickers=tickers)
	except KeyboardInterrupt:
		logger.info("Realtime stream stopped by user.")
		sys.exit(0)
	except Exception as e:
		logger.exception(f"Realtime stream error: {e}")
		sys.exit(3)


if __name__ == "__main__":
	main()
