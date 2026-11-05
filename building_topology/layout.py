"""Геометрия плана: участки стен, стены, проёмы, балки и колонны, layout.json (разделы 3, 6, 7, 9 ТЗ).

Координаты внутри — шаги сетки в системе модели. В layout.json они сдвинуты так, что
минимальные наружные грани дома в нуле, и переведены в миллиметры.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import constants as C
from .model import FAR_CLEARANCE, NEAR_CLEARANCE, STRIP, Plan
from .problem import Building, Room

HORIZONTAL = ("north", "south")
OUTWARD = {"north": 1, "south": -1, "east": 1, "west": -1}
PAD = C.EXTERIOR_WALL + 2
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class Rect:
    id: str
    floor: int
    x: int
    y: int
    w: int
    h: int

    @property
    def cell(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w + C.INTERIOR_WALL, self.y + self.h + C.INTERIOR_WALL

    def along(self, facade: str) -> tuple[int, int]:
        return (self.x, self.x + self.w) if facade in HORIZONTAL else (self.y, self.y + self.h)

    def face(self, facade: str) -> int:
        """Координата внутренней грани со стороны фасада."""
        return {"north": self.y + self.h, "south": self.y, "east": self.x + self.w, "west": self.x}[facade]

    def band(self, facade: str) -> int:
        """Ряд клеток сразу за ячейкой комнаты со стороны фасада."""
        x0, y0, x1, y1 = self.cell
        return {"north": y1, "south": y0 - 1, "east": x1, "west": x0 - 1}[facade]


@dataclass
class Run:
    """Участок стены: максимальный отрезок грани одного вида."""
    room: str
    facade: str
    lo: int
    hi: int
    kind: str
    other: str | None
    near: int = C.OPENING_END_GAP
    far: int = C.OPENING_END_GAP


@dataclass
class Wall:
    floor: int
    kind: str
    facade: str
    across: tuple[int, int]
    lo: int
    hi: int
    rooms: tuple[str, ...]

    @property
    def thickness(self) -> int:
        return self.across[1] - self.across[0]


@dataclass
class Opening:
    floor: int
    kind: str
    width: int
    wall: int
    lo: int
    rooms: tuple[str, ...]
    swing: str
    facade: str | None = None
    entrance: bool = False

    @property
    def hi(self) -> int:
        return self.lo + self.width


@dataclass(frozen=True)
class Beam:
    """Балка в перекрытии над этажом floor: держит наружную стену этажа выше между двумя опорами.

    Концы — клетки оси стены; опора на конце — стена этажа floor или колонна.
    """
    floor: int
    start: tuple[int, int]
    end: tuple[int, int]

    @property
    def length(self) -> int:
        return abs(self.end[0] - self.start[0]) + abs(self.end[1] - self.start[1])


@dataclass(frozen=True)
class Column:
    """Колонна на этаже floor; cell — клетка её центра, сечение COLUMN×COLUMN."""
    floor: int
    cell: tuple[int, int]


@dataclass
class Layout:
    building: Building
    rooms: dict[str, Rect]
    runs: dict[str, list[Run]]
    walls: list[Wall]
    openings: list[Opening]
    strips: dict[int, list[tuple[int, int, int, int]]] = field(default_factory=dict)
    beams: list[Beam] = field(default_factory=list)
    columns: list[Column] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def floor_rooms(self, floor: int) -> list[Rect]:
        return [r for r in self.rooms.values() if r.floor == floor]

    @property
    def origin(self) -> tuple[int, int]:
        """Минимальные наружные грани дома в шагах сетки."""
        xs = [r.x for r in self.rooms.values()]
        ys = [r.y for r in self.rooms.values()]
        return min(xs) - C.EXTERIOR_WALL, min(ys) - C.EXTERIOR_WALL

    def grids(self) -> dict[int, Grid]:
        """Растры всех этажей в общей рамке с полем PAD вокруг дома."""
        xs = [v for g in self.rooms.values() for v in (g.x, g.x + g.w)]
        ys = [v for g in self.rooms.values() for v in (g.y, g.y + g.h)]
        ox, oy = min(xs) - PAD, min(ys) - PAD
        nx, ny = max(xs) + PAD - ox, max(ys) + PAD - oy
        return {f: Grid(self, f, ox, oy, nx, ny) for f in range(1, self.building.floors + 1)}


class Grid:
    """Клетки одного этажа в сдвинутых координатах: индекс [x, y].

    footprint — дом со всеми стенами, solid — стены (наружные, внутренние и несущие).
    """

    def __init__(self, layout: Layout, floor: int, ox: int, oy: int, nx: int, ny: int):
        self.ox, self.oy = ox, oy
        self.cells = np.zeros((nx, ny), dtype=np.int16)
        self.inner = np.zeros((nx, ny), dtype=bool)
        self.strip = np.zeros((nx, ny), dtype=bool)
        for r in layout.floor_rooms(floor):
            x0, y0, x1, y1 = r.cell
            self.cells[self.slice(x0, y0, x1, y1)] += 1
            self.inner[self.slice(r.x, r.y, r.x + r.w, r.y + r.h)] = True
        for rect in layout.strips.get(floor, ()):
            self.strip[self.slice(*rect)] = True
        occupied = (self.cells > 0) | self.strip
        self.footprint = np.zeros_like(occupied)
        for dx in range(-C.EXTERIOR_WALL, C.EXTERIOR_WALL):
            for dy in range(-C.EXTERIOR_WALL, C.EXTERIOR_WALL):
                self.footprint |= np.roll(np.roll(occupied, dx, 0), dy, 1)
        self.solid = self.footprint & ~self.inner

    def slice(self, x0: int, y0: int, x1: int, y1: int):
        return slice(x0 - self.ox, x1 - self.ox), slice(y0 - self.oy, y1 - self.oy)

    def index(self, cell: tuple[int, int]) -> tuple[int, int]:
        return cell[0] - self.ox, cell[1] - self.oy


def covering(rooms: list[Rect], i: int, j: int) -> Rect | None:
    for r in rooms:
        x0, y0, x1, y1 = r.cell
        if x0 <= i < x1 and y0 <= j < y1:
            return r
    return None


def _inside(rects, i: int, j: int) -> bool:
    return any(x0 <= i < x1 and y0 <= j < y1 for x0, y0, x1, y1 in rects)


def face_runs(room: Rect, facade: str, floor: list[Rect], strips=()) -> list[Run]:
    """Делит грань комнаты на участки: внешний, общий с соседом, несущий или узел стен между ними.

    Несущий участок — общий с соседом через полосу несущей стены.
    """
    lo, hi = room.along(facade)
    band = room.band(facade)
    others = [r for r in floor if r.id != room.id]

    def point(u: int, line: int) -> tuple[int, int]:
        return (u, line) if facade in HORIZONTAL else (line, u)

    def probe(u: int, line: int = band) -> Rect | None:
        return covering(others, *point(u, line))

    def classify(u: int) -> tuple[str, str | None]:
        s = probe(u)
        kind = "shared"
        if s is None and _inside(strips, *point(u, band)):
            s = probe(u, band + OUTWARD[facade] * STRIP)
            kind = "bearing"
            if s is None:
                return "junction", None
        if s is None:
            return "exterior", None
        s_lo, s_hi = s.along(facade)
        return (kind, s.id) if s_lo <= u < s_hi else ("junction", None)

    runs: list[Run] = []
    for u in range(lo, hi):
        kind, other = classify(u)
        if runs and runs[-1].kind == kind and runs[-1].other == other and runs[-1].hi == u:
            runs[-1].hi = u + 1
        else:
            runs.append(Run(room.id, facade, u, u + 1, kind, other))
    for run in runs:
        if run.kind == "exterior":
            run.near = NEAR_CLEARANCE if probe(run.lo - 1) else C.OPENING_END_GAP
            run.far = FAR_CLEARANCE if probe(run.hi) else C.OPENING_END_GAP
    return runs


def _spread(widths: list[int], lo: int, hi: int) -> list[int]:
    """Равномерно расставляет проёмы на отрезке [lo, hi) с простенками не меньше OPENING_GAP."""
    n = len(widths)
    free = hi - lo - sum(widths)
    if free >= (n + 1) * C.OPENING_GAP:
        gaps = [free // (n + 1)] * (n + 1)
        for k in range(free - sum(gaps)):
            gaps[1 + k % max(n - 1, 1) if n > 1 else 0] += 1
    else:
        inner = [free // (n - 1)] * (n - 1) if n > 1 else []
        gaps = [0] + inner + [0]
    starts, at = [], lo + gaps[0]
    for k, w in enumerate(widths):
        starts.append(at)
        at += w + gaps[k + 1]
    return starts


def build(building: Building, plan: Plan) -> Layout:
    rooms = {rid: Rect(rid, building.room(rid).floor, *xywh) for rid, xywh in plan.rooms.items()}
    by_floor = {f: [r for r in rooms.values() if r.floor == f] for f in range(1, building.floors + 1)}
    runs = {rid: [run for facade in C.FACADES
                  for run in face_runs(r, facade, by_floor[r.floor], plan.strips.get(r.floor, ()))]
            for rid, r in rooms.items()}

    walls: list[Wall] = []
    wall_of: dict[tuple, int] = {}
    for rid, room in rooms.items():
        for run in runs[rid]:
            face = room.face(run.facade)
            if run.kind == "exterior":
                across = sorted((face, face + OUTWARD[run.facade] * C.EXTERIOR_WALL))
                wall_of[(rid, run.facade, run.lo)] = len(walls)
                walls.append(Wall(room.floor, "exterior", run.facade, tuple(across), run.lo, run.hi, (rid,)))
            elif run.kind in ("shared", "bearing") and run.facade in ("north", "east"):
                kind, thickness = ("interior", C.INTERIOR_WALL) if run.kind == "shared" else ("bearing", C.BEARING_WALL)
                key = (rid, run.other)
                wall_of[key] = wall_of[(run.other, rid)] = len(walls)
                walls.append(Wall(room.floor, kind, run.facade, (face, face + thickness), run.lo, run.hi, key))
            elif run.kind == "junction":
                across = tuple(sorted((face, face + OUTWARD[run.facade] * C.INTERIOR_WALL)))
                walls.append(Wall(room.floor, "interior", run.facade, across, run.lo, run.hi, (rid,)))

    openings: list[Opening] = []
    groups: dict[tuple, list] = {}
    for p in plan.placements:
        room = rooms[p.opening.room]
        run = next(r for r in runs[room.id] if r.facade == p.facade and r.kind == "exterior"
                   and r.lo <= p.start and p.start + p.opening.width <= r.hi)
        groups.setdefault((room.id, p.facade, run.lo), []).append(p)
    for (rid, facade, run_lo), placed in groups.items():
        run = next(r for r in runs[rid] if r.facade == facade and r.lo == run_lo)
        placed.sort(key=lambda p: p.start)
        starts = _spread([p.opening.width for p in placed], run.lo + run.near, run.hi - run.far)
        index = wall_of[(rid, facade, run_lo)]
        for p, lo in zip(placed, starts):
            o = p.opening
            openings.append(Opening(rooms[rid].floor, o.kind, o.width, index, lo, (rid,), rid, facade, o.entrance))

    clearances = [box for o in openings if o.kind == "door" for box in door_clearance(rooms, walls, o)]
    for a, c in building.doors:
        index = wall_of[(a, c)]
        wall = walls[index]
        ra, rc = building.room(a), building.room(c)
        swing = max((ra.target, a), (rc.target, c))[1]
        middle = wall.lo + (wall.hi - wall.lo - C.INTERIOR_DOOR) // 2
        candidates = dict.fromkeys((middle, wall.lo + C.OPENING_END_GAP,
                                    wall.hi - C.OPENING_END_GAP - C.INTERIOR_DOOR))
        options = [Opening(rooms[a].floor, "door", C.INTERIOR_DOOR, index, lo, wall.rooms, swing) for lo in candidates]
        door = min(options, key=lambda o: sum(_common(box, other) for box in door_clearance(rooms, walls, o)
                                               for other in clearances if other[0] == o.floor))
        openings.append(door)
        clearances.extend(door_clearance(rooms, walls, door))

    layout = Layout(building, rooms, runs, walls, openings, dict(plan.strips), stats=dict(plan.stats))
    layout.beams, layout.columns = structure(layout)
    return layout


def _erode(mask: np.ndarray) -> np.ndarray:
    """Клетки маски, у которых все восемь соседей тоже в маске."""
    out = mask.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            out &= np.roll(np.roll(mask, dx, 0), dy, 1)
    return out


def axis_loops(footprint: np.ndarray) -> list[tuple[list[tuple[int, int]], np.ndarray]]:
    """Оси наружных стен этажа: замкнутые цепочки клеток на глубине 2 от края контура.

    Возвращает цепочки индексов клеток и маску клеток стены поперёк оси (глубины 1 и 3).
    """
    e1 = _erode(footprint)
    e2 = _erode(e1)
    e3 = _erode(e2)
    ring = e1 & ~e2
    across = (footprint & ~e1) | (e2 & ~e3)
    left = {tuple(int(v) for v in c) for c in np.argwhere(ring)}
    loops = []
    while left:
        start = min(left)
        path, prev, cur = [start], None, start
        left.discard(start)
        while True:
            steps = [(cur[0] + dx, cur[1] + dy) for dx, dy in NEIGHBOURS]
            steps = [c for c in steps if ring[c] and c != prev]
            if len(path) > 2 and start in steps:
                break
            fresh = [c for c in steps if c in left]
            if not fresh:
                break
            prev, cur = cur, min(fresh)
            path.append(cur)
            left.discard(cur)
        loops.append((path, across))
    return loops


def structure(layout: Layout) -> tuple[list[Beam], list[Column]]:
    """Балки под висящими участками наружных стен и колонны под ними (раздел 7 ТЗ).

    Клетка оси стены опёрта, если под ней и ещё под одной клеткой стены поперёк оси лежит стена
    этажа ниже. Подряд идущие неопёртые клетки несёт балка от опоры до опоры: опора — опёртая
    клетка оси или колонна. Колонна ставится там, где балка поворачивает за угол контура, и
    через равные шаги, если пролёт длиннее допустимого. Колонна, под которой нет стены, повторяется
    на этаже ниже.
    """
    grids = layout.grids()
    span = layout.building.beam_span
    beams: list[Beam] = []
    columns: dict[tuple[int, tuple[int, int]], Column] = {}
    for floor in range(2, layout.building.floors + 1):
        below = grids[floor - 1]

        def cell(c: tuple[int, int]) -> tuple[int, int]:
            return c[0] + below.ox, c[1] + below.oy

        for path, across in axis_loops(grids[floor].footprint):

            def held(c: tuple[int, int]) -> bool:
                return bool(below.solid[c]) and any(across[c[0] + dx, c[1] + dy] and below.solid[c[0] + dx, c[1] + dy]
                                                    for dx, dy in NEIGHBOURS)

            n = len(path)
            rest = [held(c) for c in path]
            if all(rest):
                continue
            step = [(path[(k + 1) % n][0] - path[k][0], path[(k + 1) % n][1] - path[k][1]) for k in range(n)]
            turn = [step[k - 1] != step[k] for k in range(n)]
            first = rest.index(True) if any(rest) else turn.index(True)
            order = list(range(first, n)) + list(range(first))
            stops = [first] + [k for k in order[1:] if rest[k] or turn[k]] + [first]
            for a, b in zip(stops, stops[1:]):
                if rest[a] and rest[b] and (b - a) % n == 1:
                    continue
                p, q = path[a], path[b]
                parts = math.ceil((abs(q[0] - p[0]) + abs(q[1] - p[1])) / span)
                points = [(p[0] + (q[0] - p[0]) * k // parts, p[1] + (q[1] - p[1]) * k // parts) for k in range(parts)]
                points.append(q)
                beams += [Beam(floor - 1, cell(u), cell(v)) for u, v in zip(points, points[1:])]
                for u in points:
                    if not held(u):
                        columns.setdefault((floor - 1, cell(u)), Column(floor - 1, cell(u)))
    for floor in range(layout.building.floors, 1, -1):
        below = grids[floor - 1]
        for col in [c for c in columns.values() if c.floor == floor]:
            if not below.solid[below.index(col.cell)]:
                columns.setdefault((floor - 1, col.cell), Column(floor - 1, col.cell))
    return beams, sorted(columns.values(), key=lambda c: (c.floor, c.cell))


OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}
DOOR_FRONT = 5


def swing_side(rooms: dict[str, Rect], walls: list[Wall], o: Opening) -> str:
    """Сторона стены, если смотреть из комнаты, куда открывается дверь."""
    wall = walls[o.wall]
    return wall.facade if o.swing == wall.rooms[0] else OPPOSITE[wall.facade]


def door_clearance(rooms: dict[str, Rect], walls: list[Wall], o: Opening) -> list[tuple]:
    """Зона открывания полотна и подход к двери с другой стороны: (этаж, x0, y0, x1, y1)."""
    side = swing_side(rooms, walls, o)
    face = rooms[o.swing].face(side)
    wall = walls[o.wall]
    zones = [sorted((face, face - OUTWARD[side] * o.width))]
    if wall.kind != "exterior":
        back = wall.across[1] if OUTWARD[side] > 0 else wall.across[0]
        zones.append(sorted((back, back + OUTWARD[side] * DOOR_FRONT)))
    out = []
    for c0, c1 in zones:
        rect = (o.lo, c0, o.hi, c1) if wall.facade in HORIZONTAL else (c0, o.lo, c1, o.hi)
        out.append((o.floor, *rect))
    return out


def _common(a: tuple, b: tuple) -> int:
    w = min(a[3], b[3]) - max(a[1], b[1])
    h = min(a[4], b[4]) - max(a[2], b[2])
    return w * h if w > 0 and h > 0 else 0


def _mm(units: float) -> int:
    return round(units * C.UNIT_MM)


def area_flag(room: Room, area: int) -> str | None:
    """`larger` или `smaller`, если площадь ушла от цели дальше нормального допуска."""
    if room.kind == "stair" or C.within_tolerance(area, room.target):
        return None
    return "larger" if area > room.target else "smaller"


def to_json(layout: Layout) -> dict:
    ox, oy = layout.origin
    b = layout.building

    def point(wall: Wall, along: float) -> list[int]:
        mid = (wall.across[0] + wall.across[1]) / 2
        if wall.facade in HORIZONTAL:
            return [_mm(along - ox), _mm(mid - oy)]
        return [_mm(mid - ox), _mm(along - oy)]

    rooms = []
    for r in b.rooms:
        g = layout.rooms[r.id]
        item = {"id": r.id, "floor": r.floor, "x": _mm(g.x - ox), "y": _mm(g.y - oy), "w": _mm(g.w), "h": _mm(g.h),
                "area_m2": round(g.w * g.h * C.CELL_M2, 2), "target_area_m2": round(r.target * C.CELL_M2, 2)}
        flag = area_flag(r, g.w * g.h)
        if flag:
            item["area_flag"] = flag
        rooms.append(item)
    walls = []
    for k, wall in enumerate(layout.walls):
        walls.append({"index": k, "floor": wall.floor, "kind": wall.kind, "thickness_mm": _mm(wall.thickness),
                      "start": point(wall, wall.lo), "end": point(wall, wall.hi), "rooms": list(wall.rooms)})
    openings = []
    for o in layout.openings:
        wall = layout.walls[o.wall]
        item = {"floor": o.floor, "kind": o.kind, "width_mm": _mm(o.width), "wall": o.wall,
                "start": point(wall, o.lo), "end": point(wall, o.hi), "rooms": list(o.rooms)}
        if o.facade:
            item["facade"] = o.facade
        if o.entrance:
            item["entrance"] = True
        openings.append(item)

    def centre(c: tuple[int, int]) -> list[int]:
        return [_mm(c[0] + 0.5 - ox), _mm(c[1] + 0.5 - oy)]

    beams = [{"floor": beam.floor, "start": centre(beam.start), "end": centre(beam.end),
              "length_mm": _mm(beam.length)} for beam in layout.beams]
    columns = [{"floor": col.floor, "center": centre(col.cell), "size_mm": _mm(C.COLUMN)} for col in layout.columns]
    return {"version": 1, "floors": b.floors, "rooms": rooms, "walls": walls, "openings": openings,
            "beams": beams, "columns": columns, "metrics": metrics(layout)}


def metrics(layout: Layout) -> dict:
    b = layout.building
    deviation = sum(abs(g.w * g.h - b.room(rid).target) for rid, g in layout.rooms.items())
    first = layout.floor_rooms(1)
    x0 = min(r.x for r in first) - C.EXTERIOR_WALL
    y0 = min(r.y for r in first) - C.EXTERIOR_WALL
    x1 = max(r.x + r.w for r in first) + C.EXTERIOR_WALL
    y1 = max(r.y + r.h for r in first) + C.EXTERIOR_WALL
    out = {"area_deviation_m2": round(deviation * C.CELL_M2, 2),
           "rooms_out_of_tolerance": sum(area_flag(b.room(rid), g.w * g.h) is not None
                                         for rid, g in layout.rooms.items()),
           "total_area_m2": round(sum(g.w * g.h for g in layout.rooms.values()) * C.CELL_M2, 2),
           "outline_area_m2": round((x1 - x0) * (y1 - y0) * C.CELL_M2, 2),
           "wall_length_m": round(sum(w.hi - w.lo for w in layout.walls) * C.UNIT_MM / 1000, 2),
           "beam_length_m": round(sum(beam.length for beam in layout.beams) * C.UNIT_MM / 1000, 2),
           "columns": len(layout.columns)}
    out.update({k: v for k, v in layout.stats.items() if k in ("outline", "seconds", "status")})
    return out
