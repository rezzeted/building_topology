"""Геометрия плана: участки стен, стены, проёмы и layout.json (разделы 3, 6, 9 ТЗ).

Координаты внутри — шаги сетки в системе модели. В layout.json они сдвинуты так, что
минимальные наружные грани дома в нуле, и переведены в миллиметры.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import constants as C
from .model import FAR_CLEARANCE, NEAR_CLEARANCE, Plan
from .problem import Building

HORIZONTAL = ("north", "south")
OUTWARD = {"north": 1, "south": -1, "east": 1, "west": -1}


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


@dataclass
class Layout:
    building: Building
    rooms: dict[str, Rect]
    runs: dict[str, list[Run]]
    walls: list[Wall]
    openings: list[Opening]
    stats: dict = field(default_factory=dict)

    def floor_rooms(self, floor: int) -> list[Rect]:
        return [r for r in self.rooms.values() if r.floor == floor]

    @property
    def origin(self) -> tuple[int, int]:
        """Минимальные наружные грани дома в шагах сетки."""
        xs = [r.x for r in self.rooms.values()]
        ys = [r.y for r in self.rooms.values()]
        return min(xs) - C.EXTERIOR_WALL, min(ys) - C.EXTERIOR_WALL


def covering(rooms: list[Rect], i: int, j: int) -> Rect | None:
    for r in rooms:
        x0, y0, x1, y1 = r.cell
        if x0 <= i < x1 and y0 <= j < y1:
            return r
    return None


def face_runs(room: Rect, facade: str, floor: list[Rect]) -> list[Run]:
    """Делит грань комнаты на участки: внешний, общий с соседом или узел стен между ними."""
    lo, hi = room.along(facade)
    band = room.band(facade)
    others = [r for r in floor if r.id != room.id]

    def probe(u: int) -> Rect | None:
        return covering(others, u, band) if facade in HORIZONTAL else covering(others, band, u)

    def classify(u: int) -> tuple[str, str | None]:
        s = probe(u)
        if s is None:
            return "exterior", None
        s_lo, s_hi = s.along(facade)
        return ("shared", s.id) if s_lo <= u < s_hi else ("junction", None)

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
    runs = {rid: [run for facade in C.FACADES for run in face_runs(r, facade, by_floor[r.floor])]
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
            elif run.kind == "shared" and run.facade in ("north", "east"):
                across = (face, face + C.INTERIOR_WALL)
                key = (rid, run.other)
                wall_of[key] = wall_of[(run.other, rid)] = len(walls)
                walls.append(Wall(room.floor, "interior", run.facade, across, run.lo, run.hi, key))
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

    return Layout(building, rooms, runs, walls, openings, dict(plan.stats))


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
    if wall.kind == "interior":
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
        rooms.append({"id": r.id, "floor": r.floor, "x": _mm(g.x - ox), "y": _mm(g.y - oy),
                      "w": _mm(g.w), "h": _mm(g.h), "area_m2": round(g.w * g.h * C.CELL_M2, 2)})
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
    return {"version": 1, "floors": b.floors, "rooms": rooms, "walls": walls, "openings": openings,
            "metrics": metrics(layout)}


def metrics(layout: Layout) -> dict:
    b = layout.building
    deviation = sum(abs(g.w * g.h - b.room(rid).target) for rid, g in layout.rooms.items())
    first = layout.floor_rooms(1)
    x0 = min(r.x for r in first) - C.EXTERIOR_WALL
    y0 = min(r.y for r in first) - C.EXTERIOR_WALL
    x1 = max(r.x + r.w for r in first) + C.EXTERIOR_WALL
    y1 = max(r.y + r.h for r in first) + C.EXTERIOR_WALL
    out = {"area_deviation_m2": round(deviation * C.CELL_M2, 2),
           "total_area_m2": round(sum(g.w * g.h for g in layout.rooms.values()) * C.CELL_M2, 2),
           "outline_area_m2": round((x1 - x0) * (y1 - y0) * C.CELL_M2, 2),
           "wall_length_m": round(sum(w.hi - w.lo for w in layout.walls) * C.UNIT_MM / 1000, 2)}
    out.update({k: v for k, v in layout.stats.items() if k in ("outline", "seconds", "status")})
    return out
