"""Независимая проверка готового плана по разделам 6 и 7 ТЗ на сетке клеток."""

from __future__ import annotations

from collections import deque

import numpy as np

from . import constants as C
from .layout import HORIZONTAL, OUTWARD, Grid, Layout, Wall, opening_conflicts, own_window
from .problem import rectangles


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
        elif (g.w, g.h) not in rectangles(r.kind, r.target, b.room_tolerance):
            errors.append(f"{r.id}: прямоугольник {g.w}×{g.h} нарушает площадь, сторону или пропорции")
    for s in b.shafts:
        rects = {(rooms[rid].x, rooms[rid].y, rooms[rid].w, rooms[rid].h) for rid in s.rooms}
        if len(rects) != 1:
            errors.append(f"шахта {s.id}: комнаты стоят в разных местах")
    total = sum(g.w * g.h for g in rooms.values())
    if not C.within_tolerance(total, b.house_area):
        errors.append(f"сумма площадей {total * C.CELL_M2:.2f} м² вне допуска дома")

    grids = layout.grids()
    for f, grid in grids.items():
        if (grid.cells > 1).any():
            errors.append(f"этаж {f}: комнаты со стенами перекрываются")
        if (grid.strip & (grid.cells > 0)).any():
            errors.append(f"этаж {f}: несущая стена заходит в ячейку комнаты")
        occupied = (grid.cells > 0) | grid.strip
        first = tuple(np.argwhere(occupied)[0])
        if not _connected(occupied, first):
            errors.append(f"этаж {f}: этаж распадается на части")
        if not _connected(~grid.footprint, (0, 0)):
            errors.append(f"этаж {f}: в контуре есть замкнутый двор")
        if f > 1 and (grid.footprint & ~grids[f - 1].footprint).any():
            errors.append(f"этаж {f}: контур выходит за контур этажа {f - 1}")

    errors += _check_walls(layout, grids)
    errors += _check_support(layout, grids)
    errors += _check_openings(layout)
    errors += _check_heaters(layout, grids)
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
            beyond = (wall.lo, band, wall.hi, band + 1) if wall.facade in HORIZONTAL else (band, wall.lo, band + 1, wall.hi)
            if (grid.cells[grid.slice(*beyond)] > 0).any() or grid.strip[grid.slice(*beyond)].any():
                errors.append(f"стена {k}: участок {room.id} {wall.facade} не внешний")
        elif len(wall.rooms) == 2:
            a, c = (rooms[rid] for rid in wall.rooms)
            gap = C.INTERIOR_WALL if wall.kind == "interior" else C.BEARING_WALL
            if c.face(_opposite(wall.facade)) - a.face(wall.facade) != gap:
                errors.append(f"стена {k}: между {a.id} и {c.id} зазор не {gap * C.UNIT_MM} мм")
            if wall.kind == "bearing":
                x0, y0, x1, y1 = wall_rect(wall)
                beyond_fringe = (x0, y0 + 1, x1, y1) if wall.facade in HORIZONTAL else (x0 + 1, y0, x1, y1)
                if not grid.strip[grid.slice(*beyond_fringe)].all():
                    errors.append(f"стена {k}: несущая стена {a.id}–{c.id} не заполнена полосой")
            lo = max(a.along(wall.facade)[0], c.along(wall.facade)[0])
            hi = min(a.along(wall.facade)[1], c.along(wall.facade)[1])
            if (lo, hi) != (wall.lo, wall.hi) or hi <= lo:
                errors.append(f"стена {k}: общий участок {a.id}–{c.id} задан неверно")
    for i, w in enumerate(exterior):
        for v in exterior[i + 1:]:
            if w.floor == v.floor and v.facade == _opposite(w.facade) and _overlap(wall_rect(w), wall_rect(v)):
                errors.append(f"этаж {w.floor}: наружные стены {w.rooms[0]} и {v.rooms[0]} сходятся в узкой выемке")
    return errors


def _check_support(layout: Layout, grids: dict[int, Grid]) -> list[str]:
    """Наружные и несущие стены верхних этажей стоят на стенах, балках или колоннах (раздел 7.13).

    Позиция стены опёрта, если стена этажа ниже лежит под средним шагом её толщины и ещё под одним.
    Иначе позицию несёт балка, если балки разрешены: её ось проходит по среднему шагу стены.
    """
    b = layout.building
    errors = []
    strict = not b.allow_beams
    if strict and (layout.beams or layout.columns):
        errors.append("в плане есть балки или колонны, а structure.allow_beams не задан")
    columns = {(c.floor, c.cell) for c in layout.columns}

    def on_beam(floor: int, cell: tuple[int, int]) -> bool:
        for beam in layout.beams:
            if beam.floor != floor:
                continue
            (x0, y0), (x1, y1) = beam.start, beam.end
            if min(x0, x1) <= cell[0] <= max(x0, x1) and min(y0, y1) <= cell[1] <= max(y0, y1):
                return True
        return False

    for k, wall in enumerate(layout.walls):
        if wall.floor == 1 or wall.kind == "interior":
            continue
        below = grids[wall.floor - 1]
        c0 = wall.across[0]
        hanging = 0
        for t in range(wall.lo, wall.hi):
            units = [(t, c) if wall.facade in HORIZONTAL else (c, t) for c in range(c0, c0 + C.EXTERIOR_WALL)]
            solid = [bool(below.solid[below.index(u)]) for u in units]
            if solid[1] and sum(solid) >= 2:
                continue
            if strict or wall.kind == "bearing" or not on_beam(wall.floor - 1, units[1]):
                hanging += 1
        if hanging:
            errors.append(f"стена {k} ({wall.kind}, этаж {wall.floor}): {hanging * C.UNIT_MM} мм без опоры")

    for beam in layout.beams:
        (x0, y0), (x1, y1) = beam.start, beam.end
        if x0 != x1 and y0 != y1:
            errors.append(f"балка {beam.start}–{beam.end}: не прямая")
        if beam.length > b.beam_span:
            errors.append(f"балка {beam.start}–{beam.end}: пролёт {beam.length * C.UNIT_MM} мм длиннее допустимого")
        grid = grids[beam.floor]
        for end in (beam.start, beam.end):
            if not grid.solid[grid.index(end)] and (beam.floor, end) not in columns:
                errors.append(f"балка {beam.start}–{beam.end}: конец {end} без опоры")
    for col in layout.columns:
        grid = grids[col.floor]
        if not grid.footprint[grid.index(col.cell)]:
            errors.append(f"колонна {col.cell} на этаже {col.floor}: вне контура этажа")
        if col.floor > 1:
            below = grids[col.floor - 1]
            if not below.solid[below.index(col.cell)] and (col.floor - 1, col.cell) not in columns:
                errors.append(f"колонна {col.cell} на этаже {col.floor}: под ней нет стены или колонны")
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
            if wall.kind == "exterior" or wall.hi - wall.lo < C.INTERIOR_DOOR + 2 * C.OPENING_END_GAP:
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
        if layout.walls[k].kind != "exterior" and len(items) > 1:
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


def _inside(inner, outer) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _interior(r) -> tuple[int, int, int, int]:
    return r.x, r.y, r.x + r.w, r.y + r.h


def _flush(rect, outer) -> bool:
    return rect[0] == outer[0] or rect[1] == outer[1] or rect[2] == outer[2] or rect[3] == outer[3]


def _check_heaters(layout: Layout, grids: dict[int, Grid]) -> list[str]:
    """Отопители, трубы и выходы (раздел 7 ТЗ, п. 16–25)."""
    b = layout.building
    errors = []
    rooms = layout.rooms
    if sorted(h.id for h in layout.heaters) != sorted(h.id for h in b.heaters):
        errors.append("отопители плана не совпадают с отопителями входа")
    specs = {h.id: h for h in b.heaters}
    floors = range(1, b.floors + 1)

    for h in layout.heaters:
        spec, r = specs[h.id], rooms[h.room]
        name = f"отопитель {h.id}"
        along, depth = spec.size
        if (h.kind, h.room, h.floor, h.flue) != (spec.kind, spec.room, spec.floor, spec.flue):
            errors.append(f"{name}: вид, комната или труба не как во входе")
        if (h.w, h.h) != ((along, depth) if h.back in HORIZONTAL else (depth, along)):
            errors.append(f"{name}: размер {h.w}×{h.h} не по таблице")
        if not _inside(h.rect, _interior(r)):
            errors.append(f"{name}: выходит из комнаты {r.id}")
        if {"north": h.rect[3], "south": h.rect[1], "east": h.rect[2], "west": h.rect[0]}[h.back] != r.face(h.back):
            errors.append(f"{name}: не стоит спиной к стене {h.back}")
        lo, hi = h.span
        run = next((u for u in layout.runs[r.id] if u.facade == h.back and u.lo <= lo and hi <= u.hi), None)
        need = ("shared", "bearing") if h.flue == "chimney" else ("exterior",)
        if run is None or run.kind not in need:
            wall = "внутренняя стена с одной комнатой" if h.flue == "chimney" else "наружная стена"
            errors.append(f"{name}: за спиной на всю ширину не {wall}")
        elif run.kind == "exterior" and (lo < run.lo + run.near - C.OPENING_END_GAP
                                         or hi > run.hi - run.far + C.OPENING_END_GAP):
            errors.append(f"{name}: заходит за наружную стену соседней комнаты во внутреннем углу фасада")
        front = h.front
        if front:
            if not _inside(front, _interior(r)):
                errors.append(f"{name}: зона перед ним выходит из комнаты")
            for rect, other in layout.solids(h.floor):
                if other != h.id and _overlap(front, rect):
                    errors.append(f"{name}: в зону перед ним заходит {other}")
        if h.kind == "gas_convector":
            r_lo, r_hi = r.along(h.back)
            if lo - r_lo < C.CONVECTOR_SIDE_GAP or r_hi - hi < C.CONVECTOR_SIDE_GAP:
                errors.append(f"{name}: ближе {C.CONVECTOR_SIDE_GAP * C.UNIT_MM} мм к боковой стене")
            if not any(own_window(h, o, layout.walls) for o in layout.openings):
                errors.append(f"{name}: не под окном своей комнаты")
    for o in layout.openings:
        under = [h.id for h in layout.heaters if own_window(h, o, layout.walls)]
        if len(under) > 1:
            errors.append(f"окно {o.rooms[0]}: под ним больше одного конвектора")
    for r in b.rooms:
        convectors = sum(h.kind == "gas_convector" for h in layout.heaters if h.room == r.id)
        area = rooms[r.id].w * rooms[r.id].h
        if convectors and area < C.CONVECTOR_ROOM_MIN:
            errors.append(f"{r.id}: с конвектором площадь меньше {C.CONVECTOR_ROOM_MIN * C.CELL_M2:g} м²")
        if convectors == 1 and area >= C.CONVECTOR_PAIR_AREA:
            errors.append(f"{r.id}: при площади от {C.CONVECTOR_PAIR_AREA * C.CELL_M2:g} м² один конвектор")

    for f in floors:
        solids = layout.solids(f)
        for i, (a, ida) in enumerate(solids):
            for c, idc in solids[i + 1:]:
                if _overlap(a, c):
                    errors.append(f"этаж {f}: {ida} и {idc} пересекаются")

    heaters = {h.id: h for h in layout.heaters}
    for c in layout.chimneys:
        h = heaters[c.heater]
        name = f"труба {c.heater}"
        cw, cd = C.CHIMNEY
        if (c.w, c.h) != ((cw, cd) if h.back in HORIZONTAL else (cd, cw)):
            errors.append(f"{name}: размер {c.w}×{c.h} не по таблице")
        lo, hi = (c.x, c.x + c.w) if h.back in HORIZONTAL else (c.y, c.y + c.h)
        if lo - h.span[0] != (C.HEATER_SIZE[h.kind][0] - cw) // 2 or hi > h.span[1]:
            errors.append(f"{name}: не посередине отопителя")
        face = rooms[h.room].face(h.back)
        if c.outside:
            near = {"north": c.y, "south": c.y + c.h, "east": c.x, "west": c.x + c.w}[h.back]
            outer = face + OUTWARD[h.back] * C.EXTERIOR_WALL
            if near != outer:
                errors.append(f"{name}: не примыкает снаружи к стене за отопителем")
            if c.floors != tuple(floors):
                errors.append(f"{name}: приставная труба не на всех этажах")
            outside = floors
        else:
            if not _inside(c.rect, h.rect) or not _flush(c.rect, h.rect):
                errors.append(f"{name}: на этаже отопителя не в его корпусе у стены")
            if c.floors != tuple(range(h.floor, c.floors[-1] + 1)):
                errors.append(f"{name}: этажи трубы идут не подряд от этажа отопителя")
            for f in c.floors[1:]:
                hosts = [r for r in layout.floor_rooms(f) if b.room(r.id).kind != "stair"
                         and _inside(c.rect, _interior(r)) and _flush(c.rect, _interior(r))]
                if len(hosts) != 1:
                    errors.append(f"{name}: на этаже {f} не стоит у стены одной комнаты")
            outside = range(c.floors[-1] + 1, b.floors + 1)
        for f in outside:
            grid = grids[f]
            if grid.footprint[grid.slice(*c.rect)].any():
                errors.append(f"{name}: на этаже {f} пересекает контур этажа")

    for o in layout.openings:
        errors += opening_conflicts(layout, o)
    gas = [h for h in layout.heaters if h.gas]
    for i, a in enumerate(gas):
        for c in gas[i + 1:]:
            if a.floor == c.floor and a.back == c.back:
                gap = C.CONVECTOR_OUTLET_GAP if a.kind == c.kind == "gas_convector" else C.OUTLET_GAP
                if abs(a.outlet - c.outlet) < gap + 1:
                    errors.append(f"выходы {a.id} и {c.id}: ближе {gap * C.UNIT_MM} мм")
    for o in layout.openings:
        if o.kind != "door":
            continue
        gas_rooms = [rid for rid in o.rooms if b.gas_room(rid)]
        if gas_rooms and o.swing in gas_rooms:
            errors.append(f"дверь {'–'.join(o.rooms)}: открывается внутрь комнаты с газовым прибором")
    return errors


def _opposite(facade: str) -> str:
    return {"north": "south", "south": "north", "east": "west", "west": "east"}[facade]
