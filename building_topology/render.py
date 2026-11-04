"""Чертёж этажа в SVG и PNG по разделу 10 ТЗ."""

from __future__ import annotations

from collections import defaultdict
from xml.sax.saxutils import escape

import cairocffi
import cairosvg

from . import constants as C
from .check import Grid
from .layout import HORIZONTAL, OUTWARD, Layout, Opening, swing_side

PX = C.PX_PER_M * C.UNIT_MM / 1000
FONT = "DejaVu Sans"
LABEL_SIZE, LABEL_MIN = 14, 8
STEP = 2.5
MARGIN, TOP, BOTTOM = 40, 90, 90
WALL_LINE, THIN_LINE = 1.4, 0.8

_measure = cairocffi.Context(cairocffi.ImageSurface(cairocffi.FORMAT_ARGB32, 1, 1))
_measure.select_font_face(FONT)


def text_width(text: str, size: float) -> float:
    _measure.set_font_size(size)
    return _measure.text_extents(text)[4]


def area_text(cells: int) -> str:
    tenths = (cells + 5) // 10
    return f"{tenths // 10},{tenths % 10} м²"


def _merge(edges: set[tuple[str, int, int]]) -> list[tuple[str, int, int, int]]:
    """Склеивает единичные отрезки (ось, линия, позиция) в отрезки (ось, линия, начало, конец)."""
    lines = defaultdict(list)
    for orient, line, pos in edges:
        lines[(orient, line)].append(pos)
    out = []
    for (orient, line), positions in sorted(lines.items()):
        positions.sort()
        start = prev = positions[0]
        for p in positions[1:] + [None]:
            if p != prev + 1:
                out.append((orient, line, start, prev + 1))
                if p is not None:
                    start = p
            if p is not None:
                prev = p
    return out


class Sheet:
    def __init__(self, layout: Layout):
        self.layout = layout
        self.ox, self.oy = layout.origin
        first = layout.floor_rooms(1)
        self.width_u = max(r.x + r.w for r in first) + C.EXTERIOR_WALL - self.ox
        self.height_u = max(r.y + r.h for r in first) + C.EXTERIOR_WALL - self.oy
        self.width = max(round(self.width_u * PX) + 2 * MARGIN, 5 * C.PX_PER_M + 2 * MARGIN + 120)

    def x(self, u: float) -> float:
        return round(MARGIN + (u - self.ox) * PX, 2)

    def y(self, u: float) -> float:
        return round(TOP + (self.height_u - (u - self.oy)) * PX, 2)

    def point(self, facade: str, along: float, across: float) -> tuple[float, float]:
        return (self.x(along), self.y(across)) if facade in HORIZONTAL else (self.x(across), self.y(along))

    def floor_svg(self, floor: int) -> str:
        layout = self.layout
        rooms = layout.floor_rooms(floor)
        openings = [o for o in layout.openings if o.floor == floor]
        legend = []
        body: list[str] = []

        cuts = set()
        for o in openings:
            wall = layout.walls[o.wall]
            orient = "h" if wall.facade in HORIZONTAL else "v"
            for line in wall.across:
                cuts.update((orient, line, p) for p in range(o.lo, o.hi))

        inner = set()
        for r in rooms:
            for y in (r.y, r.y + r.h):
                inner.update(("h", y, p) for p in range(r.x, r.x + r.w))
            for x in (r.x, r.x + r.w):
                inner.update(("v", x, p) for p in range(r.y, r.y + r.h))
        grid = Grid(layout, floor, self.ox - 5, self.oy - 5,
                    self.width_u + 10, self.height_u + 10)
        mask = grid.footprint
        outer = set()
        for i, j in zip(*(mask[:, 1:] != mask[:, :-1]).nonzero()):
            outer.add(("h", grid.oy + j + 1, grid.ox + i))
        for i, j in zip(*(mask[1:, :] != mask[:-1, :]).nonzero()):
            outer.add(("v", grid.ox + i + 1, grid.oy + j))

        for r in rooms:
            body.append(f'<rect x="{self.x(r.x)}" y="{self.y(r.y + r.h)}" width="{r.w * PX:g}" '
                        f'height="{r.h * PX:g}" fill="#fff"/>')
        lines = [self._segment(seg) for seg in _merge((inner | outer) - cuts)]
        body.append(f'<g stroke="#000" stroke-width="{WALL_LINE}" stroke-linecap="square">{"".join(lines)}</g>')

        marks = []
        for o in openings:
            marks.extend(self._opening(o))
        for r in rooms:
            if layout.building.room(r.id).kind == "stair":
                marks.extend(self._stair(r))
        body.append(f'<g stroke="#000" stroke-width="{THIN_LINE}" fill="none">{"".join(marks)}</g>')

        for r in rooms:
            body.extend(self._label(r, legend))

        height = TOP + round(self.height_u * PX) + BOTTOM + 18 * len(legend)
        head = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{height}" '
                f'viewBox="0 0 {self.width} {height}" font-family="{FONT}">',
                f'<rect width="{self.width}" height="{height}" fill="#fff"/>',
                f'<text x="{MARGIN}" y="34" font-size="20">Этаж {floor}</text>']
        tail = self._north() + self._scale(TOP + round(self.height_u * PX) + 40)
        for k, line in enumerate(legend):
            tail.append(f'<text x="{MARGIN}" y="{height - BOTTOM + 70 + 18 * k}" font-size="12">'
                        f'{escape(line)}</text>')
        return "\n".join(head + body + tail + ["</svg>"]) + "\n"

    def _segment(self, seg) -> str:
        orient, line, a, b = seg
        if orient == "h":
            return f'<line x1="{self.x(a)}" y1="{self.y(line)}" x2="{self.x(b)}" y2="{self.y(line)}"/>'
        return f'<line x1="{self.x(line)}" y1="{self.y(a)}" x2="{self.x(line)}" y2="{self.y(b)}"/>'

    def _line(self, facade, a0, c0, a1, c1) -> str:
        (x1, y1), (x2, y2) = self.point(facade, a0, c0), self.point(facade, a1, c1)
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'

    def _opening(self, o: Opening) -> list[str]:
        wall = self.layout.walls[o.wall]
        f = wall.facade
        c0, c1 = wall.across
        out = [self._line(f, o.lo, c0, o.lo, c1), self._line(f, o.hi, c0, o.hi, c1)]
        if o.kind == "window":
            mid = (c0 + c1) / 2
            out.append(self._line(f, o.lo, mid, o.hi, mid))
            return out
        side = swing_side(self.layout.rooms, self.layout.walls, o)
        face = self.layout.rooms[o.swing].face(side)
        inward = -OUTWARD[side]
        hinge = self.point(f, o.lo, face)
        leaf = self.point(f, o.lo, face + inward * o.width)
        close = self.point(f, o.hi, face)
        v1 = (leaf[0] - hinge[0], leaf[1] - hinge[1])
        v2 = (close[0] - hinge[0], close[1] - hinge[1])
        sweep = 1 if v1[0] * v2[1] - v1[1] * v2[0] > 0 else 0
        r = round(o.width * PX, 2)
        out.append(f'<line x1="{hinge[0]}" y1="{hinge[1]}" x2="{leaf[0]}" y2="{leaf[1]}"/>')
        out.append(f'<path d="M {leaf[0]} {leaf[1]} A {r} {r} 0 0 {sweep} {close[0]} {close[1]}"/>')
        return out

    def _stair(self, r) -> list[str]:
        b = self.layout.building
        room = b.room(r.id)
        shaft = next(s for s in b.shafts if s.id == room.shaft)
        top = room.floor == shaft.to_floor
        gap = C.OPENING_END_GAP
        along_x = r.w >= r.h
        lo, hi = (r.x + gap, r.x + r.w - gap) if along_x else (r.y + gap, r.y + r.h - gap)
        c0, c1 = (r.y + gap, r.y + r.h - gap) if along_x else (r.x + gap, r.x + r.w - gap)
        facade = "north" if along_x else "east"
        out = [self._line(facade, lo, c0, hi, c0), self._line(facade, lo, c1, hi, c1)]
        p = lo
        while p <= hi + 1e-9:
            out.append(self._line(facade, p, c0, p, c1))
            p += STEP
        mid = (c0 + c1) / 2
        start, end = (hi - STEP / 2, lo + STEP / 2) if top else (lo + STEP / 2, hi - STEP / 2)
        out.append(self._line(facade, start, mid, end, mid))
        tip = self.point(facade, end, mid)
        sign = 1 if end > start else -1
        left = self.point(facade, end - sign * 1.2, mid + 0.6)
        right = self.point(facade, end - sign * 1.2, mid - 0.6)
        out.append(f'<path d="M {tip[0]} {tip[1]} L {left[0]} {left[1]} L {right[0]} {right[1]} Z" fill="#000"/>')
        return out

    def _label(self, r, legend: list[str]) -> list[str]:
        room = self.layout.building.room(r.id)
        lines = [room.name, area_text(r.w * r.h)]
        width, height = r.w * PX - 4, r.h * PX - 4
        cx, cy = self.x(r.x + r.w / 2), self.y(r.y + r.h / 2)
        backed = room.kind == "stair"
        for size in range(LABEL_SIZE, LABEL_MIN - 1, -1):
            if max(text_width(t, size) for t in lines) <= width and 2.4 * size <= height:
                return self._text_block(lines, size, cx, cy, backed)
        legend.append(f"{room.id} — {room.name}, {lines[1]}")
        if text_width(room.id, LABEL_MIN) <= width and 1.2 * LABEL_MIN <= height:
            return self._text_block([room.id], LABEL_MIN, cx, cy, backed)
        return []

    def _text_block(self, lines: list[str], size: int, cx: float, cy: float, backed: bool) -> list[str]:
        """Подпись по центру; на лестнице под ней белая подложка, чтобы ступени не перечёркивали текст."""
        step = 1.2 * size
        first = cy - (len(lines) - 1) * step / 2 + 0.35 * size
        out = []
        if backed:
            w = max(text_width(t, size) for t in lines) + 4
            h = len(lines) * step
            out.append(f'<rect x="{cx - w / 2:.2f}" y="{cy - h / 2:.2f}" width="{w:.2f}" height="{h:.2f}" fill="#fff"/>')
        for k, t in enumerate(lines):
            out.append(f'<text x="{cx}" y="{first + k * step:.2f}" font-size="{size}" '
                       f'text-anchor="middle">{escape(t)}</text>')
        return out

    def _north(self) -> list[str]:
        x = self.width - MARGIN - 10
        return [f'<path d="M {x} 22 L {x + 9} 62 L {x} 54 L {x - 9} 62 Z" fill="#000"/>',
                f'<text x="{x}" y="80" font-size="14" text-anchor="middle">С</text>']

    def _scale(self, y: float) -> list[str]:
        out = []
        for k in range(5):
            fill = "#000" if k % 2 == 0 else "#fff"
            out.append(f'<rect x="{MARGIN + k * C.PX_PER_M}" y="{y}" width="{C.PX_PER_M}" height="6" '
                       f'fill="{fill}" stroke="#000" stroke-width="1"/>')
        for k in range(6):
            label = f"{k} м" if k == 5 else str(k)
            out.append(f'<text x="{MARGIN + k * C.PX_PER_M}" y="{y + 22}" font-size="12" '
                       f'text-anchor="middle">{label}</text>')
        out.append(f'<text x="{MARGIN + 5 * C.PX_PER_M + 40}" y="{y + 8}" font-size="14">1:100</text>')
        return out


def render(layout: Layout) -> dict[int, tuple[str, bytes]]:
    """SVG и PNG для каждого этажа."""
    sheet = Sheet(layout)
    out = {}
    for floor in range(1, layout.building.floors + 1):
        svg = sheet.floor_svg(floor)
        out[floor] = (svg, cairosvg.svg2png(bytestring=svg.encode("utf-8")))
    return out
