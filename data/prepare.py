"""Downloads the training text: "Indian Fairy Tales" by Joseph Jacobs (1892), from Project
Gutenberg (ebook #7128). It is in the public domain. The Gutenberg header and licence footer
are removed, so only the book text is used."""

import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/7128/pg7128.txt"
OUT = Path(__file__).parent / "input.txt"


def main():
    text = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8")
    start = text.index("*** START OF")
    start = text.index("\n", start) + 1
    end = text.index("*** END OF")
    OUT.write_text(text[start:end].strip() + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(text[start:end]):,} characters)")


if __name__ == "__main__":
    main()
