"""Чтение входа и проверки раздела 5 ТЗ.

Все длины внутри программы — в шагах сетки (100 мм), площади — в клетках (0,01 м²).
"""

from __future__ import annotations

import json
import math
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from . import constants as C

ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")


class InputError(Exception):
    pass


@dataclass(frozen=True)
class WindowGroup:
    count: int
    width: int
    facade: str | None


@dataclass(frozen=True)
class ExteriorDoor:
    facade: str
    width: int
    entrance: bool


@dataclass
class Room:
    id: str
    name: str
    kind: str
    floor: int
    target: int
    windows: list[WindowGroup]
    exterior_doors: list[ExteriorDoor]
    shapes: list[tuple[int, int]] = field(default_factory=list)
    shaft: str | None = None

    @property
    def opening_length(self) -> int:
        return sum(g.count * g.width for g in self.windows) + sum(d.width for d in self.exterior_doors)


@dataclass
class Shaft:
    id: str
    from_floor: int
    to_floor: int
    width: int
    depth: int
    rooms: list[str]


@dataclass
class Building:
    floors: int
    house_area: int
    rooms: list[Room]
    doors: list[tuple[str, str]]
    shafts: list[Shaft]

    def __post_init__(self) -> None:
        self.by_id = {r.id: r for r in self.rooms}

    def room(self, room_id: str) -> Room:
        return self.by_id[room_id]

    def floor_rooms(self, floor: int) -> list[Room]:
        return [r for r in self.rooms if r.floor == floor]

    @property
    def entrance_room(self) -> Room:
        return next(r for r in self.rooms if any(d.entrance for d in r.exterior_doors))


def rectangles(kind: str, target: int) -> list[tuple[int, int]]:
    """Все прямоугольники сетки (w, h), допустимые для комнаты по площади, сторонам и пропорциям."""
    lo = math.ceil(target * (100 - C.AREA_TOLERANCE_PCT) / 100)
    hi = target * (100 + C.AREA_TOLERANCE_PCT) // 100
    side, aspect = C.MIN_SIDE[kind], C.MAX_ASPECT[kind]
    out = []
    for w in range(side, hi // side + 1):
        h_lo = max(side, math.ceil(lo / w), math.ceil(w / aspect))
        h_hi = min(hi // w, w * aspect)
        for h in range(h_lo, h_hi + 1):
            if C.within_tolerance(w * h, target):
                out.append((w, h))
    return out


def load(path: str | Path) -> Building:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"не удалось прочитать вход: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError(f"вход не является JSON: {exc}") from exc
    return parse(data)


def parse(data) -> Building:
    building = _parse_schema(data)
    _check_semantics(building)
    return building


def _fields(value, where: str, required: set[str], optional: set[str] = frozenset()) -> dict:
    if not isinstance(value, dict):
        raise InputError(f"{where}: ожидается объект")
    unknown = sorted(set(value) - required - optional)
    if unknown:
        raise InputError(f"{where}: неизвестное поле {unknown[0]!r}")
    missing = sorted(required - set(value))
    if missing:
        raise InputError(f"{where}: нет поля {missing[0]!r}")
    return value


def _int(value, where: str, lo: int | None = None, hi: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise InputError(f"{where}: ожидается целое число")
    if lo is not None and value < lo or hi is not None and value > hi:
        bounds = f"от {lo}" + (f" до {hi}" if hi is not None else "")
        raise InputError(f"{where}: значение {value} вне диапазона {bounds}")
    return value


def _length(value, where: str, minimum_mm: int) -> int:
    mm = _int(value, where)
    if mm % C.UNIT_MM:
        raise InputError(f"{where}: {mm} мм не кратно шагу сетки {C.UNIT_MM} мм")
    if mm < minimum_mm:
        raise InputError(f"{where}: {mm} мм меньше минимума {minimum_mm} мм")
    return mm // C.UNIT_MM


def _area(value, where: str) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise InputError(f"{where}: ожидается число")
    if value <= 0:
        raise InputError(f"{where}: площадь должна быть положительной")
    cells = round(value / C.CELL_M2)
    if abs(value / C.CELL_M2 - cells) > 1e-6:
        raise InputError(f"{where}: площадь {value} не кратна 0,01 м²")
    return cells


def _identifier(value, where: str) -> str:
    if not isinstance(value, str) or not ID_RE.match(value):
        raise InputError(f"{where}: идентификатор {value!r} не подходит под [A-Za-z][A-Za-z0-9_-]*")
    return value


def _facade(value, where: str) -> str:
    if value not in C.FACADES:
        raise InputError(f"{where}: фасад {value!r} не из {', '.join(C.FACADES)}")
    return value


def _list(value, where: str) -> list:
    if not isinstance(value, list):
        raise InputError(f"{where}: ожидается список")
    return value


def _parse_schema(data) -> Building:
    top = _fields(data, "вход", {"version", "floors", "house_area_m2", "rooms"}, {"doors", "stairs"})
    if _int(top["version"], "version") != 1:
        raise InputError(f"version: поддерживается только 1, получено {top['version']}")
    floors = _int(top["floors"], "floors", 1, C.MAX_FLOORS)
    house_area = _area(top["house_area_m2"], "house_area_m2")

    raw_rooms = _list(top["rooms"], "rooms")
    if not 1 <= len(raw_rooms) <= C.MAX_ROOMS:
        raise InputError(f"rooms: комнат {len(raw_rooms)}, допустимо от 1 до {C.MAX_ROOMS}")

    ids: set[str] = set()
    rooms = []
    for i, raw in enumerate(raw_rooms):
        where = f"rooms[{i}]"
        r = _fields(raw, where, {"id", "kind", "floor", "area_m2"}, {"name", "windows", "exterior_doors"})
        rid = _identifier(r["id"], f"{where}.id")
        where = f"комната {rid}"
        if rid in ids:
            raise InputError(f"{where}: идентификатор повторяется")
        ids.add(rid)
        name = r.get("name", rid)
        if not isinstance(name, str) or not name.strip():
            raise InputError(f"{where}: name должно быть непустой строкой")
        kind = r["kind"]
        if kind not in C.KINDS:
            raise InputError(f"{where}: kind {kind!r} не из {', '.join(C.KINDS)}")
        floor = _int(r["floor"], f"{where}.floor", 1, floors)
        target = _area(r["area_m2"], f"{where}.area_m2")

        windows = []
        for j, g in enumerate(_list(r.get("windows", []), f"{where}.windows")):
            gw = f"{where}.windows[{j}]"
            g = _fields(g, gw, {"count"}, {"width_mm", "facade"})
            count = _int(g["count"], f"{gw}.count", 1)
            width = _length(g.get("width_mm", C.WINDOW_DEFAULT * C.UNIT_MM), f"{gw}.width_mm",
                            C.WINDOW_MIN * C.UNIT_MM)
            facade = _facade(g["facade"], f"{gw}.facade") if "facade" in g else None
            windows.append(WindowGroup(count, width, facade))

        ext_doors = []
        for j, d in enumerate(_list(r.get("exterior_doors", []), f"{where}.exterior_doors")):
            dw = f"{where}.exterior_doors[{j}]"
            d = _fields(d, dw, {"facade"}, {"width_mm", "entrance"})
            facade = _facade(d["facade"], f"{dw}.facade")
            width = _length(d.get("width_mm", C.EXTERIOR_DOOR_DEFAULT * C.UNIT_MM), f"{dw}.width_mm",
                            C.EXTERIOR_DOOR_MIN * C.UNIT_MM)
            entrance = d.get("entrance", False)
            if not isinstance(entrance, bool):
                raise InputError(f"{dw}.entrance: ожидается true или false")
            ext_doors.append(ExteriorDoor(facade, width, entrance))

        rooms.append(Room(rid, name, kind, floor, target, windows, ext_doors))

    by_id = {r.id: r for r in rooms}
    doors = []
    pairs: set[frozenset[str]] = set()
    for i, raw in enumerate(_list(top.get("doors", []), "doors")):
        where = f"doors[{i}]"
        d = _fields(raw, where, {"a", "b"})
        a, b = d["a"], d["b"]
        for end in (a, b):
            if end not in by_id:
                raise InputError(f"{where}: нет комнаты {end!r}")
        if a == b:
            raise InputError(f"{where}: петля {a}–{a}")
        if by_id[a].floor != by_id[b].floor:
            raise InputError(f"{where}: комнаты {a} и {b} на разных этажах")
        key = frozenset((a, b))
        if key in pairs:
            raise InputError(f"{where}: пара {a}–{b} повторяется")
        pairs.add(key)
        doors.append((a, b))

    shafts = []
    shaft_ids: set[str] = set()
    for i, raw in enumerate(_list(top.get("stairs", []), "stairs")):
        where = f"stairs[{i}]"
        s = _fields(raw, where, {"id", "from_floor", "to_floor", "width_mm", "depth_mm", "rooms"})
        sid = _identifier(s["id"], f"{where}.id")
        where = f"шахта {sid}"
        if sid in shaft_ids or sid in ids:
            raise InputError(f"{where}: идентификатор повторяется")
        shaft_ids.add(sid)
        lo = _int(s["from_floor"], f"{where}.from_floor", 1, floors)
        hi = _int(s["to_floor"], f"{where}.to_floor", 1, floors)
        width = _length(s["width_mm"], f"{where}.width_mm", C.STAIR_MIN_SIDE * C.UNIT_MM)
        depth = _length(s["depth_mm"], f"{where}.depth_mm", C.STAIR_MIN_SIDE * C.UNIT_MM)
        members = _list(s["rooms"], f"{where}.rooms")
        shafts.append(Shaft(sid, lo, hi, width, depth, members))

    return Building(floors, house_area, rooms, doors, shafts)


def _check_semantics(b: Building) -> None:
    # 5.2
    total = sum(r.target for r in b.rooms)
    if not C.within_tolerance(total, b.house_area):
        raise InputError(
            f"сумма площадей комнат {total * C.CELL_M2:.2f} м² отличается от house_area_m2 "
            f"{b.house_area * C.CELL_M2:.2f} м² больше чем на {C.AREA_TOLERANCE_PCT}%")

    # 5.4, 5.5
    if b.floors == 1 and b.shafts:
        raise InputError("при floors = 1 список stairs должен быть пуст")
    for s in b.shafts:
        where = f"шахта {s.id}"
        if s.from_floor >= s.to_floor:
            raise InputError(f"{where}: from_floor должен быть меньше to_floor")
        if len(s.rooms) != s.to_floor - s.from_floor + 1:
            raise InputError(f"{where}: нужно по одной комнате на каждый этаж от {s.from_floor} до {s.to_floor}")
        for floor, rid in zip(range(s.from_floor, s.to_floor + 1), s.rooms):
            if rid not in b.by_id:
                raise InputError(f"{where}: нет комнаты {rid!r}")
            r = b.by_id[rid]
            if r.kind != "stair":
                raise InputError(f"{where}: комната {rid} должна иметь kind = stair")
            if r.floor != floor:
                raise InputError(f"{where}: комната {rid} должна быть на этаже {floor}, а она на {r.floor}")
            if r.shaft is not None:
                raise InputError(f"{where}: комната {rid} уже входит в шахту {r.shaft}")
            if r.target != s.width * s.depth:
                raise InputError(
                    f"{where}: площадь комнаты {rid} {r.target * C.CELL_M2:.2f} м² не равна площади шахты "
                    f"{s.width * s.depth * C.CELL_M2:.2f} м²")
            r.shaft = s.id
            r.shapes = [(s.width, s.depth)]
    for r in b.rooms:
        if r.kind == "stair" and r.shaft is None:
            raise InputError(f"комната {r.id}: лестничная комната не входит ни в одну шахту")

    reached, queue = {1}, deque([1])
    while queue:
        floor = queue.popleft()
        for s in b.shafts:
            if s.from_floor <= floor <= s.to_floor:
                for f in range(s.from_floor, s.to_floor + 1):
                    if f not in reached:
                        reached.add(f)
                        queue.append(f)
    missing = sorted(set(range(1, b.floors + 1)) - reached)
    if missing:
        raise InputError(f"этаж {missing[0]} не достижим по шахтам от этажа 1")

    # 5.6, 5.7
    entrances = [r for r in b.rooms for d in r.exterior_doors if d.entrance]
    if len(entrances) != 1:
        raise InputError(f"входов в дом {len(entrances)}, должен быть ровно один (entrance: true)")
    for r in b.rooms:
        if r.exterior_doors and r.floor != 1:
            raise InputError(f"комната {r.id}: наружные двери допустимы только на этаже 1")
        if r.exterior_doors and r.kind == "stair":
            raise InputError(f"комната {r.id}: у лестничной комнаты не может быть наружных дверей")

    # 5.3, 5.9
    for r in b.rooms:
        if r.kind != "stair":
            r.shapes = rectangles(r.kind, r.target)
            if not r.shapes:
                raise InputError(
                    f"комната {r.id}: нет прямоугольника площадью {r.target * C.CELL_M2:.2f} м² ±{C.AREA_TOLERANCE_PCT}% "
                    f"со стороной не меньше {C.MIN_SIDE[r.kind] * C.UNIT_MM} мм и пропорцией не больше "
                    f"{C.MAX_ASPECT[r.kind]}")
        perimeter = max(2 * (w + h) for w, h in r.shapes)
        if r.opening_length > perimeter:
            raise InputError(
                f"комната {r.id}: проёмы общей длиной {r.opening_length * C.UNIT_MM} мм не помещаются "
                f"ни в один допустимый периметр (максимум {perimeter * C.UNIT_MM} мм)")

    # 5.8
    adjacency: dict[str, set[str]] = {r.id: set() for r in b.rooms}
    for a, c in b.doors:
        adjacency[a].add(c)
        adjacency[c].add(a)
    for s in b.shafts:
        for a, c in zip(s.rooms, s.rooms[1:]):
            adjacency[a].add(c)
            adjacency[c].add(a)
    start = entrances[0].id
    seen, queue = {start}, deque([start])
    while queue:
        for n in adjacency[queue.popleft()]:
            if n not in seen:
                seen.add(n)
                queue.append(n)
    lost = [r.id for r in b.rooms if r.id not in seen]
    if lost:
        raise InputError(f"комната {lost[0]} не достижима от входа ({start}) по дверям и лестницам")
