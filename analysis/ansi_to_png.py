#!/usr/bin/env python3
"""Render coloured terminal output (ANSI SGR codes) to a PNG.

The README's terminal captures used to be hand-taken screenshots, which meant
they could silently drift away from the run the rest of the numbers came from.
This renders them from the binaries' own output instead, so they regenerate
with everything else.

Usage (from core/, after training):
    script -qc "./build/play --report --model ./models/final_agent" /dev/null \\
        | python ../analysis/ansi_to_png.py --until "Not scored" \\
            -o ../docs/images/strategy_chart.png

`script` gives the binary a pseudo-terminal, which is what makes it emit colour.
"""

import argparse
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# GitHub dark palette, so the captures sit naturally in a README.
BACKGROUND = "#0d1117"
FOREGROUND = "#c9d1d9"
COLOURS = {
    31: "#f85149",  # red
    32: "#3fb950",  # green
    33: "#d29922",  # yellow
    34: "#58a6ff",  # blue
    35: "#bc8cff",  # magenta
    36: "#39c5cf",  # cyan
    37: FOREGROUND,
}
DIM = "#8b949e"

SGR = re.compile(r"\x1b\[([0-9;]*)m")
OTHER_ESCAPES = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

FONT = "DejaVu Sans Mono"
CHAR_WIDTH_EM = 0.602  # DejaVu Sans Mono advance width
LINE_HEIGHT_EM = 1.35


def parse(text: str) -> list[list[tuple[str, str, bool]]]:
    """Split text into lines of (segment, colour, bold) runs."""
    lines = []
    colour, bold, dim = FOREGROUND, False, False
    for raw in text.replace("\r\n", "\n").replace("\r", "").split("\n"):
        runs = []
        pos = 0
        for m in SGR.finditer(raw):
            if m.start() > pos:
                runs.append((raw[pos : m.start()], DIM if dim else colour, bold))
            codes = [int(c) for c in m.group(1).split(";") if c] or [0]
            for code in codes:
                if code == 0:
                    colour, bold, dim = FOREGROUND, False, False
                elif code == 1:
                    bold = True
                elif code == 2:
                    dim = True
                elif code in COLOURS:
                    colour = COLOURS[code]
            pos = m.end()
        if pos < len(raw):
            runs.append((raw[pos:], DIM if dim else colour, bold))
        runs = [(OTHER_ESCAPES.sub("", s), c, b) for s, c, b in runs]
        lines.append([r for r in runs if r[0]])
    return lines


def plain(line) -> str:
    return "".join(text for text, _, _ in line)


def trim(lines, head, until=None):
    while lines and not lines[0]:
        lines.pop(0)
    if until:
        for i, line in enumerate(lines):
            if re.search(until, plain(line)):
                lines = lines[: i + 1]
                break
    if head:
        lines = lines[:head]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def render(lines, output: str, font_size: float, pad_chars: float = 2.0) -> None:
    cols = max((sum(len(s) for s, _, _ in line) for line in lines), default=1)
    char_w = CHAR_WIDTH_EM * font_size / 72.0  # inches
    line_h = LINE_HEIGHT_EM * font_size / 72.0
    pad = pad_chars * char_w
    width = cols * char_w + 2 * pad
    height = len(lines) * line_h + 2 * pad

    fig = plt.figure(figsize=(width, height), facecolor=BACKGROUND)
    for row, line in enumerate(lines):
        y = 1.0 - (pad + (row + 0.8) * line_h) / height
        col = 0
        for text, colour, bold in line:
            x = (pad + col * char_w) / width
            fig.text(x, y, text, fontfamily=FONT, fontsize=font_size,
                     color=colour, fontweight="bold" if bold else "normal")
            col += len(text)
    fig.savefig(output, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", nargs="?", help="ANSI text file (default: stdin)")
    parser.add_argument("-o", "--output", required=True, help="PNG path")
    parser.add_argument("--head", type=int, default=0,
                        help="Keep only the first N lines (after leading blanks)")
    parser.add_argument("--until", metavar="REGEX",
                        help="Stop after the first line matching REGEX")
    parser.add_argument("--font-size", type=float, default=13.0)
    args = parser.parse_args()

    text = open(args.input, encoding="utf-8").read() if args.input else sys.stdin.read()
    lines = trim(parse(text), args.head, args.until)
    if not lines:
        print("No input to render", file=sys.stderr)
        return 1
    render(lines, args.output, args.font_size)
    print(f"Wrote {args.output} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
