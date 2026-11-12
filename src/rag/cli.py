import argparse
import json
from pathlib import Path
from app.operations.process import (
    process_text_file,
    embed_text_file,
)
from app.inference.local_inference_model import LocalInferenceProvider
from app.database.factory import get_database
from app.database.migrations import migrate

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("db:create")

    chat_parser = commands.add_parser("chat")

    chat_parser.add_argument(
        "--model",
        type=str
    )

    chat_parser.add_argument(
        "--query",
        type=str
    )

    embed_parser = commands.add_parser("embed")

    embed_parser.add_argument("content")

    embed_parser.add_argument(
        "--model",
        type=str
    )

    embed_parser.add_argument(
        "--chunk_size",
        type=int
    )

    embed_parser.add_argument(
        "--stride",
        type=int
    )

    embed_parser.add_argument(
        "--batch_size",
        type=int
    )

    embed_parser.add_argument(
        "--output",
        type=str
    )

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
    elif args.command == "chat":
        query = args.query
        model = args.model
        
        inference_provider = LocalInferenceProvider(
            model_path=model
        )

        result = inference_provider.infer(query)

        inference_provider.close()

        print(result)

    elif args.command == "embed":
        text_file = args.content
        chunk_size = args.chunk_size
        stride = args.stride
        batch_size = args.batch_size
        output = args.output
        model = args.model

        result = embed_text_file(
            text_file,
            chunk_size,
            stride,
            model,
            batch_size,
            None
        )

        serialized = json.dumps(result, indent=2)

        if args.output:
            Path(args.output).write_text(serialized + "\n", encoding="utf-8")
        else:
            print(serialized)

    elif args.command == "db:create":
        database = get_database()
        database.create_database()
        print("Database initialized")

    elif args.command == "db:migrate":
        database = get_database()
        completed = migrate(database)

        if not completed:
            print("No pending migrations")

        else:
            for version in completed:
                print(f"Applied migration: {version}")

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
