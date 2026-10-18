#!/usr/bin/env python3
"""Create a private .env once. Never print the generated API key."""
from pathlib import Path
import secrets


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / ".env"
    template = (root / ".env.example").read_text(encoding="utf-8")
    key = secrets.token_urlsafe(32)
    content = template.replace("CASHWATCH_API_KEY=\n", f"CASHWATCH_API_KEY={key}\n", 1)
    try:
        with destination.open("x", encoding="utf-8") as file:
            destination.chmod(0o600)
            file.write(content)
    except FileExistsError:
        raise SystemExit(".env already exists; it was not overwritten.")
    print("Created .env. Read the local file to use the key in the UI; do not commit it.")


if __name__ == "__main__":
    main()

