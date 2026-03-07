#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_summary(path: Path):
    return json.loads(path.read_text())


def metric(summary, key, default=0.0):
    for section in ("timing", "multimodal", "async"):
        if key in summary.get(section, {}):
            return summary[section][key].get("mean", default)
    return default


def main():
    parser = argparse.ArgumentParser(description="Combine multiple multimodal profile summaries into one markdown table")
    parser.add_argument("--input", action="append", required=True, help="Label=path/to/summary.json")
    parser.add_argument("--output", required=True, help="Markdown output path")
    args = parser.parse_args()

    rows = []
    for item in args.input:
        label, path = item.split("=", 1)
        summary = load_summary(Path(path))
        rows.append({
            "label": label,
            "step_s": metric(summary, "timing_s/step"),
            "gen_s": metric(summary, "timing_s/gen"),
            "reward_s": metric(summary, "timing_s/reward"),
            "old_log_prob_s": metric(summary, "timing_s/old_log_prob"),
            "actor_s": metric(summary, "timing_s/update_actor"),
            "queue_wait_s": metric(summary, "timing_s/queue_wait"),
            "payload_mb": metric(summary, "multimodal/payload_mb"),
            "buffer": summary.get("buffer_bottleneck", "n/a"),
        })

    lines = [
        "# Multimodal RL Benchmark Summary",
        "",
        "| Label | Step (s) | Gen (s) | Reward (s) | Old Log Prob (s) | Actor Update (s) | Queue Wait (s) | Payload (MB) | Buffer Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['step_s']:.4f} | {row['gen_s']:.4f} | {row['reward_s']:.4f} | "
            f"{row['old_log_prob_s']:.4f} | {row['actor_s']:.4f} | {row['queue_wait_s']:.4f} | "
            f"{row['payload_mb']:.4f} | {row['buffer']} |"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote suite summary to {output_path}")


if __name__ == "__main__":
    main()
