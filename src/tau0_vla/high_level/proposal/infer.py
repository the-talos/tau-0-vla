"""Run proposal inference on a portable JSONL manifest."""
import argparse
import json
from pathlib import Path

from .data import read_samples


def build_parser() -> argparse.ArgumentParser:
    """CLI surface, split out so tests can pin flags without running the model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--device",
        default=None,
        help="xpu | cuda | cpu, with an optional index (xpu:1). Auto-detects XPU > CUDA > CPU when omitted.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    rows = read_samples(args.input)
    if args.output.exists():
        raise FileExistsError(args.output)
    from .runtime import Proposal
    policy = Proposal(args.model, args.adapter, args.device, args.max_new_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for row in rows:
            handle.write(json.dumps({"id": row["id"], **policy.predict(row)}, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
