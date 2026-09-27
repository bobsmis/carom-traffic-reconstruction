from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.pipeline.processor import VideoRoadProcessor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local video-to-metric-road reconstruction.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ua_detrac_clean_intersection.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "ua-detrac-clean-intersection")
    args = parser.parse_args()
    processor = VideoRoadProcessor(ROOT, args.config.resolve(), args.output.resolve())
    audit = processor.run()
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
