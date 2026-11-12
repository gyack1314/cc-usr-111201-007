"""资料检查、校准分析与合成资料生成的命令行入口，不改变原文件。"""
import argparse
import json

from .io import load
from .domain import summarize
from .calibration import analyze
from . import synthetic


def _print(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect", help="原始资料摘要与基础统计")
    inspect_p.add_argument("file")

    calibrate_p = sub.add_parser("calibrate", help="有界熵校准与 jackknife 方差")
    calibrate_p.add_argument("file")

    gen_p = sub.add_parser("generate-materials", help="写出三套校准核对合成资料")
    gen_p.add_argument("directory")

    args = parser.parse_args()
    if args.command == "inspect":
        value, digest = load(args.file)
        _print({"source_sha256": digest, "summary": summarize(value)})
    elif args.command == "calibrate":
        value, digest = load(args.file)
        report = analyze(value)
        report = {"source_sha256": digest, **report}
        _print(report)
    else:
        from pathlib import Path

        directory = Path(args.directory)
        directory.mkdir(parents=True, exist_ok=True)
        _print({"written": [str(p) for p in synthetic.write(directory)]})


if __name__ == "__main__":
    main()
