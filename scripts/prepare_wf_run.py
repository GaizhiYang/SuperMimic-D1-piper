"""Allocate fresh, matching training/reference directories for a WF pipeline."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def prepare_run(config_path, refs_base) -> Path:
    config = json.loads(Path(config_path).read_text())
    output_base = Path(config["output"])
    refs_base = Path(refs_base)
    if output_base.resolve() == refs_base.resolve():
        raise ValueError("training output and reference directory must have different base paths")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    collision = 0
    while True:
        suffix = stamp if collision == 0 else f"{stamp}_{collision}"
        output = output_base.with_name(f"{output_base.name}_{suffix}")
        references = refs_base.with_name(f"{refs_base.name}_{suffix}")
        try:
            output.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            collision += 1
            continue
        try:
            references.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            output.rmdir()  # Only the empty directory just allocated by this call.
            collision += 1
            continue
        break
    config["output"] = str(output)
    config.setdefault("demonstrations", {})["directory"] = str(references / "library")
    # A new stage-1 run starts fresh, even if its template came from a resumed
    # trainer's config.json. Explicit dynamics warm starts remain configured.
    config["resume"] = None
    snapshot = output / "pipeline_config.json"
    snapshot.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--refs-base", required=True)
    args = parser.parse_args()
    print(prepare_run(args.config, args.refs_base))


if __name__ == "__main__":
    main()
