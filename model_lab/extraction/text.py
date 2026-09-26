"""Deterministic HTML-to-readable-text conversion with block offsets for traceability."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

TEXT_VERSION = "readable-text-v1"
SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}
BLOCK_TAGS = {
    "p", "div", "section", "article", "header", "footer", "nav", "main", "aside",
    "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "li", "ul", "ol",
    "table", "tr", "td", "th", "dt", "dd", "dl", "br", "hr", "title", "form",
}
KIND = {
    **{f"h{n}": "heading" for n in range(1, 7)}, "p": "paragraph", "pre": "code",
    "li": "list-item", "dd": "paragraph", "dt": "term", "title": "title",
}
VOID_TAGS = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "wbr", "source"}


@dataclass(frozen=True)
class Block:
    kind: str
    tag: str
    start: int
    end: int
    in_nav: bool
    classes: tuple[str, ...]


@dataclass(frozen=True)
class Readable:
    text: str
    blocks: tuple[Block, ...]

    def block_text(self, block: Block) -> str:
        return self.text[block.start : block.end]


class _Converter(HTMLParser):
    def __init__(self, skip_tags: set[str]) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_tags = skip_tags
        self.parts: list[str] = []
        self.length = 0
        self.skip = 0
        self.pre = 0
        self.stack: list[tuple[str, int, tuple[str, ...], bool]] = []
        self.blocks: list[Block] = []
        self.pending_space = False

    def _emit(self, text: str) -> None:
        self.parts.append(text)
        self.length += len(text)

    def _break(self) -> None:
        self.pending_space = False
        if self.parts and not self.parts[-1].endswith("\n"):
            self._emit("\n")

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            if tag not in VOID_TAGS:
                self.skip += 1
            return
        if self.skip:
            return
        if tag in BLOCK_TAGS:
            self._break()
        if tag == "text":  # separate SVG labels, e.g. syntax-diagram words
            self.pending_space = True
        if tag in VOID_TAGS:
            return
        classes = tuple(sorted((dict(attrs).get("class") or "").split()))
        nav = tag == "nav" or any(entry[3] for entry in self.stack) or bool(
            {"menu", "nav", "navigation", "search", "nosearch"} & set(classes)
        )
        if tag == "pre":
            self.pre += 1
        self.stack.append((tag, self.length, classes, nav))

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip or tag in VOID_TAGS:
            return
        # Close up to the matching tag; tolerate unclosed inner elements.
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                break
        else:
            return
        while len(self.stack) > index:
            name, start, classes, nav = self.stack.pop()
            if name == "pre":
                self.pre = max(0, self.pre - 1)
            if name in KIND and self.length > start:
                self.blocks.append(Block(KIND[name], name, start, self.length, nav, classes))
        if tag in BLOCK_TAGS:
            self._break()

    def handle_data(self, data):
        if self.skip or not data:
            return
        if self.pre:
            if self.pending_space:
                self._emit(" ")
                self.pending_space = False
            self._emit(data)
            return
        collapsed = re.sub(r"\s+", " ", data)
        if collapsed.startswith(" "):
            self.pending_space = True
            collapsed = collapsed[1:]
        if not collapsed:
            return
        if self.pending_space and self.parts and not self.parts[-1].endswith(("\n", " ")):
            self._emit(" ")
        self.pending_space = collapsed.endswith(" ")
        self._emit(collapsed.rstrip(" "))


def readable(html: str, *, include_svg_text: bool = False) -> Readable:
    """Readable text; include_svg_text also keeps labels drawn inside inline SVG diagrams."""
    converter = _Converter(SKIP_TAGS - {"svg"} if include_svg_text else SKIP_TAGS)
    converter.feed(html)
    converter.close()
    text = "".join(converter.parts)
    trimmed = []
    for block in converter.blocks:
        # Trim surrounding whitespace so offsets cover exactly the visible text.
        segment = text[block.start : block.end]
        start = block.start + len(segment) - len(segment.lstrip())
        end = block.end - (len(segment) - len(segment.rstrip()))
        if end > start:
            trimmed.append(Block(block.kind, block.tag, start, end, block.in_nav, block.classes))
    return Readable(text, tuple(sorted(trimmed, key=lambda b: (b.start, -b.end))))
