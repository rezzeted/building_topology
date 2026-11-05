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
    use: str = "other"

    @property
    def opening_length(self) -> int:
        return sum(g.count * g.width for g in self.windows) + sum(d.width for d in self.exterior_doors)

    @property
    def window_count(self) -> int:
        return sum(g.count for g in self.windows)


@dataclass(frozen=True)
class Heater:
    id: str
    kind: str
    room: str
    floor: int
    flue: str

    @property
    def size(self) -> tuple[int, int]:
        """Ширина вдоль стены за спиной отопителя и глубина в комнату."""
        return C.HEATER_SIZE[self.kind]

    @property
    def front(self) -> int:
        return C.FRONT_ZONE.get(self.kind, 0)

    @property
    def gas(self) -> bool:
        return self.kind in C.GAS_KINDS


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
    allow_beams: bool = False
    beam_span: int = C.BEAM_SPAN_DEFAULT
    room_tolerance: int = C.AREA_TOLERANCE_PCT
    heaters: list[Heater] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.by_id = {r.id: r for r in self.rooms}

    def room(self, room_id: str) -> Room:
        return self.by_id[room_id]

    def floor_rooms(self, floor: int) -> list[Room]:
        return [r for r in self.rooms if r.floor == floor]

    def room_heaters(self, room_id: str) -> list[Heater]:
        return [h for h in self.heaters if h.room == room_id]

    def gas_room(self, room_id: str) -> bool:
        """В комнате стоит газовый прибор: дверь из неё открывается наружу (СП 402, п. 5.11)."""
        return any(h.gas for h in self.room_heaters(room_id))

    @property
    def entrance_room(self) -> Room:
        return next(r for r in self.rooms if any(d.entrance for d in r.exterior_doors))


def rectangles(kind: str, target: int, pct: int = C.AREA_TOLERANCE_PCT) -> list[tuple[int, int]]:
    """Все прямоугольники сетки (w, h), допустимые для комнаты по площади, сторонам и пропорциям."""
    lo = math.ceil(target * (100 - pct) / 100)
    hi = target * (100 + pct) // 100
    side, aspect = C.MIN_SIDE[kind], C.MAX_ASPECT[kind]
    out = []
    for w in range(side, hi // side + 1):
        h_lo = max(side, math.ceil(lo / w), math.ceil(w / aspect))
        h_hi = min(hi // w, w * aspect)
        for h in range(h_lo, h_hi + 1):
            if C.within_tolerance(w * h, target, pct):
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
    top = _fields(data, "вход", {"version", "floors", "house_area_m2", "rooms"},
                  {"doors", "stairs", "structure", "room_area_tolerance_pct", "heaters"})
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
        r = _fields(raw, where, {"id", "kind", "floor", "area_m2"}, {"name", "use", "windows", "exterior_doors"})
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
        use = r.get("use", "other")
        if use not in C.USES:
            raise InputError(f"{where}: use {use!r} не из {', '.join(C.USES)}")
        if kind == "stair" and "use" in r:
            raise InputError(f"{where}: у лестничной комнаты use не задаётся")

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

        rooms.append(Room(rid, name, kind, floor, target, windows, ext_doors, use=use))

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

    structure = _fields(top.get("structure", {}), "structure", set(), {"allow_beams", "max_beam_span_mm"})
    beams = structure.get("allow_beams", False)
    if not isinstance(beams, bool):
        raise InputError("structure.allow_beams: ожидается true или false")
    span = _length(structure.get("max_beam_span_mm", C.BEAM_SPAN_DEFAULT * C.UNIT_MM),
                   "structure.max_beam_span_mm", C.BEAM_SPAN_MIN * C.UNIT_MM)
    tolerance = _int(top.get("room_area_tolerance_pct", C.AREA_TOLERANCE_PCT), "room_area_tolerance_pct",
                     1, C.ROOM_TOLERANCE_MAX_PCT)

    heaters = []
    for i, raw in enumerate(_list(top.get("heaters", []), "heaters")):
        where = f"heaters[{i}]"
        h = _fields(raw, where, {"id", "kind", "room"}, {"flue"})
        hid = _identifier(h["id"], f"{where}.id")
        where = f"отопитель {hid}"
        if hid in ids or hid in shaft_ids or any(o.id == hid for o in heaters):
            raise InputError(f"{where}: идентификатор повторяется")
        kind = h["kind"]
        if kind not in C.HEATER_KINDS:
            raise InputError(f"{where}: kind {kind!r} не из {', '.join(C.HEATER_KINDS)}")
        room = by_id.get(h["room"])
        if room is None:
            raise InputError(f"{where}: нет комнаты {h['room']!r}")
        flue = h.get("flue", C.FLUES[kind][0])
        if flue not in C.FLUES[kind]:
            raise InputError(f"{where}: flue {flue!r} для {kind} не из {', '.join(C.FLUES[kind])}")
        heaters.append(Heater(hid, kind, room.id, room.floor, flue))

    return Building(floors, house_area, rooms, doors, shafts, beams, span, tolerance, heaters)


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
            r.shapes = rectangles(r.kind, r.target, b.room_tolerance)
            if not r.shapes:
                raise InputError(
                    f"комната {r.id}: нет прямоугольника площадью {r.target * C.CELL_M2:.2f} м² ±{b.room_tolerance}% "
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

    _check_heaters(b)


def heater_fits(heater: Heater, w: int, h: int) -> bool:
    """Отопитель с зоной перед ним помещается в комнату w×h спиной к одной из стен."""
    along, depth = heater.size
    if heater.kind == "gas_convector":
        along += 2 * C.CONVECTOR_SIDE_GAP
    depth += heater.front
    return w >= along and h >= depth or h >= along and w >= depth


def _check_heaters(b: Building) -> None:
    # 5.10–5.14
    for h in b.heaters:
        r = b.room(h.room)
        where = f"отопитель {h.id}"
        if r.kind == "stair":
            raise InputError(f"{where}: в лестничной комнате {r.id} отопитель не ставится")
        if h.kind in ("gas_boiler", "solid_boiler") and r.floor != 1:
            raise InputError(f"{where}: котёл ставится на этаже 1 (СП 55.13330.2016, п. 7.13)")
        if h.kind == "gas_boiler":
            if r.use not in ("kitchen", "boiler_room"):
                raise InputError(f"{where}: газовый котёл ставится в кухне или котельной, а у комнаты {r.id} "
                                 f"use = {r.use} (СП 402.1325800.2018, п. 5.5)")
            if b.floors > C.COAXIAL_MAX_FLOORS:
                raise InputError(f"{where}: выход через стену допустим в доме не выше {C.COAXIAL_MAX_FLOORS} этажей "
                                 f"(СП 402.1325800.2018, п. Г.16)")
            if not r.windows:
                raise InputError(f"{where}: в комнате {r.id} с газовым котлом нужно окно (СП 402.1325800.2018, п. 5.10)")
        if h.kind == "solid_boiler" and r.use != "boiler_room":
            raise InputError(f"{where}: твердотопливный котёл ставится в котельной (use = boiler_room), "
                             f"а у комнаты {r.id} use = {r.use}")
        if h.kind == "gas_convector":
            if b.floors > C.CONVECTOR_MAX_FLOORS:
                raise InputError(f"{where}: газовые конвекторы допустимы в доме не выше {C.CONVECTOR_MAX_FLOORS} "
                                 f"этажей (СП 402.1325800.2018, п. Е.1)")
            if r.use in ("vestibule", "bath", "wc"):
                raise InputError(f"{where}: конвектор не ставится в тамбуре, туалете и ванной, а у комнаты {r.id} "
                                 f"use = {r.use} (СП 402.1325800.2018, п. Е.1)")
        if not any(heater_fits(h, w, d) for w, d in r.shapes):
            raise InputError(f"{where}: ни в один допустимый прямоугольник комнаты {r.id} не помещается отопитель "
                             f"{h.size[0] * C.UNIT_MM}×{h.size[1] * C.UNIT_MM} мм с зоной "
                             f"{h.front * C.UNIT_MM} мм перед ним")

    # Прямоугольники комнаты с отопителями сужаются до тех, где выполнены правила отопителей.
    for r in b.rooms:
        heaters = b.room_heaters(r.id)
        if not heaters:
            continue
        if sum(h.kind == "gas_boiler" for h in heaters) > C.BOILERS_PER_ROOM:
            raise InputError(f"комната {r.id}: больше {C.BOILERS_PER_ROOM} газовых котлов "
                             f"(СП 402.1325800.2018, п. 5.7)")
        shapes = [s for s in r.shapes if all(heater_fits(h, *s) for h in heaters)]
        if not shapes:
            raise InputError(f"комната {r.id}: ни в один допустимый прямоугольник не помещаются все её отопители "
                             f"с зонами перед ними")
        convectors = sum(h.kind == "gas_convector" for h in heaters)
        if convectors:
            if convectors > r.window_count:
                raise InputError(f"комната {r.id}: конвекторов {convectors}, окон {r.window_count}; каждый конвектор "
                                 f"стоит под своим окном (СП 402.1325800.2018, п. Е.3)")
            shapes = [(w, d) for w, d in shapes if w * d >= C.CONVECTOR_ROOM_MIN]
            if not shapes:
                raise InputError(f"комната {r.id}: с конвектором площадь не меньше "
                                 f"{C.CONVECTOR_ROOM_MIN * C.CELL_M2:g} м² (СП 402.1325800.2018, п. Е.1)")
            if convectors < 2:
                shapes = [(w, d) for w, d in shapes if w * d < C.CONVECTOR_PAIR_AREA]
                if not shapes:
                    raise InputError(f"комната {r.id}: при площади от {C.CONVECTOR_PAIR_AREA * C.CELL_M2:g} м² нужно "
                                     f"не меньше двух конвекторов (СП 402.1325800.2018, п. Е.2)")
        r.shapes = shapes

    for a, c in b.doors:
        if b.gas_room(a) and b.gas_room(c):
            raise InputError(f"дверь {a}–{c}: в обеих комнатах газовые приборы, а дверь из такой комнаты "
                             f"открывается наружу (СП 402.1325800.2018, п. 5.11)")
