"""Независимая проверка готового плана по разделам 6 и 7 ТЗ на сетке клеток."""

from __future__ import annotations

from collections import deque

import numpy as np

from . import constants as C
from .layout import HORIZONTAL, Layout, Wall
from .problem import rectangles

PAD = C.EXTERIOR_WALL + 2


class Grid:
    """Клетки одного этажа в сдвинутых координатах: индекс [x, y]."""

    def __init__(self, layout: Layout, floor: int, ox: int, oy: int, nx: int, ny: int):
        self.ox, self.oy = ox, oy
        self.cells = np.zeros((nx, ny), dtype=np.int16)
        self.inner = np.zeros((nx, ny), dtype=bool)
        for r in layout.floor_rooms(floor):
            x0, y0, x1, y1 = r.cell
            self.cells[self.slice(x0, y0, x1, y1)] += 1
            self.inner[self.slice(r.x, r.y, r.x + r.w, r.y + r.h)] = True
        occupied = self.cells > 0
        self.footprint = np.zeros_like(occupied)
        for dx in range(-C.EXTERIOR_WALL, C.EXTERIOR_WALL):
            for dy in range(-C.EXTERIOR_WALL, C.EXTERIOR_WALL):
                self.footprint |= np.roll(np.roll(occupied, dx, 0), dy, 1)

    def slice(self, x0: int, y0: int, x1: int, y1: int):
        return slice(x0 - self.ox, x1 - self.ox), slice(y0 - self.oy, y1 - self.oy)


def _connected(mask: np.ndarray, start: tuple[int, int]) -> bool:
    seen = np.zeros_like(mask)
    seen[start] = True
    queue = deque([start])
    while queue:
        i, j = queue.popleft()
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and mask[a, b] and not seen[a, b]:
                seen[a, b] = True
                queue.append((a, b))
    return bool((seen == mask).all())


def wall_rect(wall: Wall, lo: int | None = None, hi: int | None = None) -> tuple[int, int, int, int]:
    lo = wall.lo if lo is None else lo
    hi = wall.hi if hi is None else hi
    c0, c1 = wall.across
    return (lo, c0, hi, c1) if wall.facade in HORIZONTAL else (c0, lo, c1, hi)


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def check(layout: Layout) -> list[str]:
    b = layout.building
    errors: list[str] = []
    rooms = layout.rooms

    for r in b.rooms:
        g = rooms[r.id]
        if r.kind == "stair":
            shaft = next(s for s in b.shafts if s.id == r.shaft)
            if (g.w, g.h) != (shaft.width, shaft.depth):
                errors.append(f"{r.id}: размер {g.w}×{g.h} не равен шахте {shaft.width}×{shaft.depth}")
        elif (g.w, g.h) not in rectangles(r.kind, r.target):
            errors.append(f"{r.id}: прямоугольник {g.w}×{g.h} нарушает площадь, сторону или пропорции")
    for s in b.shafts:
        rects = {(rooms[rid].x, rooms[rid].y, rooms[rid].w, rooms[rid].h) for rid in s.rooms}
        if len(rects) != 1:
            errors.append(f"шахта {s.id}: комнаты стоят в разных местах")
    total = sum(g.w * g.h for g in rooms.values())
    if not C.within_tolerance(total, b.house_area):
        errors.append(f"сумма площадей {total * C.CELL_M2:.2f} м² вне допуска дома")

    xs = [v for g in rooms.values() for v in (g.x, g.x + g.w)]
    ys = [v for g in rooms.values() for v in (g.y, g.y + g.h)]
    ox, oy = min(xs) - PAD, min(ys) - PAD
    nx, ny = max(xs) + PAD - ox, max(ys) + PAD - oy
    grids = {f: Grid(layout, f, ox, oy, nx, ny) for f in range(1, b.floors + 1)}
    for f, grid in grids.items():
        if (grid.cells > 1).any():
            errors.append(f"этаж {f}: комнаты со стенами перекрываются")
        occupied = grid.cells > 0
        first = tuple(np.argwhere(occupied)[0])
        if not _connected(occupied, first):
            errors.append(f"этаж {f}: этаж распадается на части")
        if not _connected(~grid.footprint, (0, 0)):
            errors.append(f"этаж {f}: в контуре есть замкнутый двор")
        if f > 1 and (grid.footprint & ~grids[f - 1].footprint).any():
            errors.append(f"этаж {f}: контур выходит за контур этажа {f - 1}")

    errors += _check_walls(layout, grids)
    errors += _check_openings(layout)
    return errors


def _check_walls(layout: Layout, grids: dict[int, Grid]) -> list[str]:
    errors = []
    rooms = layout.rooms
    exterior = [w for w in layout.walls if w.kind == "exterior"]
    for k, wall in enumerate(layout.walls):
        grid = grids[wall.floor]
        if wall.kind == "exterior":
            if grid.inner[grid.slice(*wall_rect(wall))].any():
                errors.append(f"стена {k}: наружная стена заходит в комнату")
            room = rooms[wall.rooms[0]]
            band = room.band(wall.facade)
            strip = (wall.lo, band, wall.hi, band + 1) if wall.facade in HORIZONTAL else (band, wall.lo, band + 1, wall.hi)
            if (grid.cells[grid.slice(*strip)] > 0).any():
                errors.append(f"стена {k}: участок {room.id} {wall.facade} не внешний")
        elif len(wall.rooms) == 2:
            a, c = (rooms[rid] for rid in wall.rooms)
            if c.face(_opposite(wall.facade)) - a.face(wall.facade) != C.INTERIOR_WALL:
                errors.append(f"стена {k}: между {a.id} и {c.id} зазор не {C.INTERIOR_WALL * C.UNIT_MM} мм")
            lo = max(a.along(wall.facade)[0], c.along(wall.facade)[0])
            hi = min(a.along(wall.facade)[1], c.along(wall.facade)[1])
            if (lo, hi) != (wall.lo, wall.hi) or hi <= lo:
                errors.append(f"стена {k}: общий участок {a.id}–{c.id} задан неверно")
    for i, w in enumerate(exterior):
        for v in exterior[i + 1:]:
            if w.floor == v.floor and v.facade == _opposite(w.facade) and _overlap(wall_rect(w), wall_rect(v)):
                errors.append(f"этаж {w.floor}: наружные стены {w.rooms[0]} и {v.rooms[0]} сходятся в узкой выемке")
    return errors


def _check_openings(layout: Layout) -> list[str]:
    b = layout.building
    errors = []
    by_wall: dict[int, list] = {}
    for o in layout.openings:
        by_wall.setdefault(o.wall, []).append(o)
        wall = layout.walls[o.wall]
        if o.lo < wall.lo + C.OPENING_END_GAP or o.hi > wall.hi - C.OPENING_END_GAP:
            errors.append(f"проём {o.kind} {'–'.join(o.rooms)}: ближе {C.UNIT_MM} мм к концу участка")
        if o.facade is None:
            if wall.kind != "interior" or wall.hi - wall.lo < C.INTERIOR_DOOR + 2 * C.OPENING_END_GAP:
                errors.append(f"дверь {'–'.join(o.rooms)}: не на общем участке нужной длины")
            continue
        if wall.kind != "exterior" or wall.rooms != o.rooms or wall.facade != o.facade:
            errors.append(f"проём {o.kind} {o.rooms[0]}: не на внешнем участке своей комнаты")
            continue
        cut = wall_rect(wall, o.lo, o.hi)
        for k, other in enumerate(layout.walls):
            if k != o.wall and other.floor == o.floor and _overlap(cut, wall_rect(other)):
                errors.append(f"проём {o.kind} {o.rooms[0]}: задевает стену {'–'.join(other.rooms)}")
    for k, items in by_wall.items():
        items.sort(key=lambda o: o.lo)
        for p, q in zip(items, items[1:]):
            if q.lo - p.hi < C.OPENING_GAP:
                errors.append(f"стена {k}: простенок между проёмами меньше {C.UNIT_MM} мм")
        if layout.walls[k].kind == "interior" and len(items) > 1:
            errors.append(f"стена {k}: больше одной двери на общем участке")

    doors = {frozenset(o.rooms) for o in layout.openings if o.kind == "door" and o.facade is None}
    for a, c in b.doors:
        if frozenset((a, c)) not in doors:
            errors.append(f"нет двери {a}–{c}")
    for r in b.rooms:
        if r.kind == "stair" and not any(r.id in pair for pair in doors):
            errors.append(f"{r.id}: у лестничной комнаты нет двери")
        placed = [o for o in layout.openings if o.facade and o.rooms == (r.id,)]
        for group in r.windows:
            fits = [o for o in placed if o.kind == "window" and o.width == group.width
                    and (group.facade is None or o.facade == group.facade)]
            if len(fits) < group.count:
                errors.append(f"{r.id}: окон {group.width * C.UNIT_MM} мм меньше {group.count}")
        for d in r.exterior_doors:
            if not any(o.kind == "door" and o.facade == d.facade and o.width == d.width
                       and o.entrance == d.entrance for o in placed):
                errors.append(f"{r.id}: нет наружной двери на фасаде {d.facade}")
        needed = sum(g.count for g in r.windows) + len(r.exterior_doors)
        if len(placed) != needed:
            errors.append(f"{r.id}: наружных проёмов {len(placed)}, по заданию {needed}")
    return errors


def _opposite(facade: str) -> str:
    return {"north": "south", "south": "north", "east": "west", "west": "east"}[facade]
