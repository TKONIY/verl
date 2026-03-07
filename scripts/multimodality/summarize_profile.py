#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from statistics import mean


def load_records(path: Path):
    records = []
    with path.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            data = item.get("data", {})
            if isinstance(data, dict):
                records.append({"step": item.get("step"), **data})
    return records


def numeric_summary(records):
    values = {}
    for record in records:
        for key, value in record.items():
            if key == "step":
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.setdefault(key, []).append(float(value))
    return {key: {"mean": mean(vals), "min": min(vals), "max": max(vals), "count": len(vals)} for key, vals in values.items() if vals}


def buffer_assessment(summary):
    step = summary.get("timing_s/step", {}).get("mean", 0.0)
    queue_wait = summary.get("timing_s/queue_wait", {}).get("mean", 0.0)
    payload_mb = summary.get("fully_async/queue_payload_mb", {}).get("mean", 0.0)
    stale = summary.get("fully_async/count/stale_samples_processed", {}).get("max", 0.0)
    queue_ratio = queue_wait / step if step else 0.0

    if queue_ratio >= 0.15 or (payload_mb >= 8.0 and queue_ratio >= 0.05):
        severity = "primary bottleneck"
    elif queue_ratio >= 0.05 or payload_mb >= 2.0 or stale > 0:
        severity = "secondary bottleneck"
    else:
        severity = "not the primary bottleneck"

    recommendation = (
        "For async multimodal RL, prefer metadata-only queue entries: prompt/response tokens, rewards, "
        "parameter version, media URI/path, image/video counts, and optional cache keys. Avoid raw media bytes "
        "or dense vision tensors in the queue unless measurements show downstream recomputation dominates. "
        "For true replay pools, default to media references plus compact training metadata; only store visual "
        "embeddings when replay reuse clearly offsets memory and transfer cost."
    )
    return severity, queue_ratio, recommendation


def write_report(output_dir: Path, label: str, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    timing = {k: v for k, v in summary.items() if k.startswith("timing_s/")}
    multimodal = {k: v for k, v in summary.items() if k.startswith("multimodal/")}
    async_metrics = {k: v for k, v in summary.items() if k.startswith("fully_async/")}
    timing = {k: v for k, v in timing.items() if not any(token in k for token in ("prompt_length", "response_length", "num_preempted"))}
    top_timing = sorted(timing.items(), key=lambda item: item[1]["mean"], reverse=True)
    severity, queue_ratio, recommendation = buffer_assessment(summary)

    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps({
        "label": label,
        "timing": timing,
        "multimodal": multimodal,
        "async": async_metrics,
        "buffer_bottleneck": severity,
        "queue_wait_ratio": queue_ratio,
        "recommendation": recommendation,
    }, indent=2))

    md_lines = [
        f"# Multimodal RL Profile Report: {label}",
        "",
        "## Summary",
        f"- Buffer assessment: **{severity}**",
        f"- Queue wait ratio: `{queue_ratio:.3f}`",
        f"- Mean step time: `{summary.get('timing_s/step', {}).get('mean', 0.0):.4f}s`",
        f"- Mean throughput: `{summary.get('perf/throughput', {}).get('mean', 0.0):.2f}` tokens/s/GPU",
        "",
        "## Top Timing Components",
    ]
    for key, stats in top_timing[:12]:
        md_lines.append(f"- `{key}`: mean `{stats['mean']:.4f}s`, max `{stats['max']:.4f}s`")
    if multimodal:
        md_lines.extend(["", "## Multimodal Payload"])
        for key, stats in sorted(multimodal.items()):
            md_lines.append(f"- `{key}`: mean `{stats['mean']:.4f}`")
    if async_metrics:
        md_lines.extend(["", "## Async Buffer Metrics"])
        for key, stats in sorted(async_metrics.items()):
            md_lines.append(f"- `{key}`: mean `{stats['mean']:.4f}`, max `{stats['max']:.4f}`")
    md_lines.extend(["", "## Replay / Buffer Recommendation", f"- {recommendation}"])
    (output_dir / "summary.md").write_text("\n".join(md_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize multimodal RL profiler metrics from VERL file logger JSONL")
    parser.add_argument("--input", required=True, help="Path to metrics JSONL produced by trainer.logger=[console,file]")
    parser.add_argument("--output-dir", required=True, help="Directory for summary artifacts")
    parser.add_argument("--label", default="run", help="Label for the generated report")
    parser.add_argument("--skip-steps", type=int, default=0, help="Ignore steps <= this value")
    args = parser.parse_args()

    records = load_records(Path(args.input))
    records = [record for record in records if int(record.get("step", 0) or 0) > args.skip_steps]
    if not records:
        raise SystemExit("No records found after filtering")
    summary = numeric_summary(records)
    write_report(Path(args.output_dir), args.label, summary)
    print(f"Wrote report to {args.output_dir}")


if __name__ == "__main__":
    main()
