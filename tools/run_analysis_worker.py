from __future__ import annotations

import argparse
import time

from backend.api.routes import worker
from backend.database import SessionLocal
from backend.services.analysis_queue import DurableAnalysisQueue


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the durable Roadwatch inference worker")
    parser.add_argument("--once", action="store_true", help="claim at most one job and exit")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    queue = DurableAnalysisQueue(SessionLocal)
    while True:
        job_id = queue.claim_next()
        if job_id is not None:
            worker.run(job_id)
        if args.once:
            return
        if job_id is None:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
