import argparse
import json
from pathlib import Path
from app.operations.process import process_text_file

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("info")

    process_parser = commands.add_parser("process")

    process_parser.add_argument(
        "medium",
        choices=["text", "text-file"]
    )

    process_parser.add_argument(
        "content"
    )

    process_parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
    )

    process_parser.add_argument(
        "--stride",
        type=int,
        default=None,
    )

    args = parser.parse_args(argv)

    if args.command == "info":
        print("rag: modular ingestion and retrieval system")
        return 0
    elif args.command == "process":
        if args.medium == "text-file":
            if args.chunk_size is None or args.stride is None:
                process_parser.error(
                    "text-file requires --chunk_size and --stride"
                )

            result = process_text_file(
                path=args.content,
                chunk_size=args.chunk_size,
                stride=args.stride
            )

            print(json.dumps(result, indent=2))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
