"""Модель CP-SAT планировки (разделы 6–8 ТЗ).

Каждая комната занимает ячейку: внутренний прямоугольник плюс один шаг стены с севера и с востока.
Ячейки соседних комнат стыкуются вплотную, тогда между их внутренними гранями ровно 100 мм.

Все этажи делят одну сетку линий X[0..n], Y[0..n] с переменными координатами. Ячейка комнаты —
прямоугольник из клеток этой сетки. Каждая клетка сетки на этаже занята ровно одной комнатой
или пуста. Пустая клетка видит край сетки по прямой через пустые клетки того же этажа, поэтому
в этаже нет дыр. Клетка, занятая на верхнем этаже, занята и этажом ниже. Сетка совпадает
с описанным прямоугольником первого этажа.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from . import constants as C
from .problem import Building, Room

GRID_EXTRA = 4
GRID_MAX = 16
SPRAWL = 2
# Пустой промежуток между ячейками одного этажа: наружные стены по его сторонам не сливаются.
GAP_MIN = 10

# Отступ проёма от ячейки соседа во внутреннем углу: наружная стена соседа занимает часть фасада.
NEAR_CLEARANCE = C.EXTERIOR_WALL - C.INTERIOR_WALL + C.OPENING_END_GAP
FAR_CLEARANCE = C.EXTERIOR_WALL + C.OPENING_END_GAP
DOOR_SPAN = C.INTERIOR_DOOR + 2 * C.OPENING_END_GAP

# Первое решение ищется с узким допуском площади: оно находится почти так же быстро,
# а оптимизация начинает с плана, близкого к целям. Доля — от времени режима.
TIGHT_PCT = 3
TIGHT_SHARE = 0.6

HORIZONTAL = ("north", "south")


class Infeasible(Exception):
    pass


class Timeout(Exception):
    pass


@dataclass(frozen=True)
class Opening:
    room: str
    kind: str
    width: int
    facade: str | None
    entrance: bool = False


@dataclass
class Placement:
    opening: Opening
    facade: str
    start: int


@dataclass
class Plan:
    rooms: dict[str, tuple[int, int, int, int]]
    placements: list[Placement]
    stats: dict = field(default_factory=dict)


@dataclass
class Span:
    """Диапазон линий сетки, который занимает ячейка комнаты по одной оси."""
    first: list
    last: list
    cover: list


@dataclass
class RoomVars:
    x: cp_model.IntVar
    y: cp_model.IntVar
    w: cp_model.IntVar | int
    h: cp_model.IntVar | int
    area: cp_model.IntVar | int
    dev: cp_model.IntVar | int
    cell_x: cp_model.IntervalVar = None
    cell_y: cp_model.IntervalVar = None
    x_end: cp_model.IntVar = None
    y_end: cp_model.IntVar = None
    cols: Span = None
    rows: Span = None


@dataclass
class OpeningVars:
    opening: Opening
    start: cp_model.IntVar
    on_facade: dict[str, cp_model.IntVar]


def openings_of(room: Room) -> list[Opening]:
    out = [Opening(room.id, "door", d.width, d.facade, d.entrance) for d in room.exterior_doors]
    for g in room.windows:
        out += [Opening(room.id, "window", g.width, g.facade)] * g.count
    return out


class PlanModel:
    def __init__(self, building: Building, rectangular: bool = False):
        self.b = building
        self.rectangular = rectangular
        self.frames: dict[int, tuple] = {}
        self.frame_spans: dict[int, tuple[Span, Span]] = {}
        self.m = cp_model.CpModel()
        self.size = self._size_bound()
        self.rooms: dict[str, RoomVars] = {}
        self.openings: list[OpeningVars] = []
        self._access: dict[tuple, cp_model.IntVar] = {}
        self._add_grid()
        self._add_rooms()
        self._add_occupancy()
        self._add_doors()
        self._add_floor_connectivity()
        self._add_openings()
        self._add_house_area()
        self._add_objectives()

    def _size_bound(self) -> int:
        biggest = 0
        for floor in range(1, self.b.floors + 1):
            cells = sum(max((w + 1) * (h + 1) for w, h in r.shapes) for r in self.b.floor_rooms(floor))
            biggest = max(biggest, cells)
        longest = max(max(max(w, h) for w, h in r.shapes) for r in self.b.rooms)
        return max(math.ceil(2 * math.sqrt(SPRAWL * biggest)), longest + 2)

    def _add_grid(self) -> None:
        m, d = self.m, self.size
        busiest = max(len(self.b.floor_rooms(f)) for f in range(1, self.b.floors + 1))
        self.n = n = min(GRID_MAX, busiest + GRID_EXTRA)
        self.xs = [0] + [m.new_int_var(1, d, f"X{i}") for i in range(1, n + 1)]
        self.ys = [0] + [m.new_int_var(1, d, f"Y{i}") for i in range(1, n + 1)]
        for lines in (self.xs, self.ys):
            for a, c in zip(lines, lines[1:]):
                m.add(c >= a + 1)

    def _span(self, start, end, lines: list, name: str) -> Span:
        m, n = self.m, self.n
        first = [m.new_bool_var(f"{name}_from{i}") for i in range(n)]
        last = [m.new_bool_var(f"{name}_to{i + 1}") for i in range(n)]
        m.add_exactly_one(first)
        m.add_exactly_one(last)
        for i in range(n):
            m.add(start == lines[i]).only_enforce_if(first[i])
            m.add(end == lines[i + 1]).only_enforce_if(last[i])
        cover = []
        for i in range(n):
            c = m.new_bool_var(f"{name}_on{i}")
            m.add(c == sum(first[:i + 1]) - sum(last[:i]))
            cover.append(c)
        return Span(first, last, cover)

    def _add_rooms(self) -> None:
        m, d = self.m, self.size
        shafts: dict[str, tuple] = {}
        for r in self.b.rooms:
            if r.kind == "stair":
                (w, h), = r.shapes
                if r.shaft not in shafts:
                    x, y = m.new_int_var(0, d, f"{r.shaft}_x"), m.new_int_var(0, d, f"{r.shaft}_y")
                    x_end, y_end = m.new_int_var(0, d, f"{r.shaft}_xe"), m.new_int_var(0, d, f"{r.shaft}_ye")
                    m.add(x_end == x + w + C.INTERIOR_WALL)
                    m.add(y_end == y + h + C.INTERIOR_WALL)
                    shafts[r.shaft] = (x, y, x_end, y_end, self._span(x, x_end, self.xs, f"{r.shaft}_c"),
                                       self._span(y, y_end, self.ys, f"{r.shaft}_r"))
                x, y, x_end, y_end, cols, rows = shafts[r.shaft]
                rv = RoomVars(x, y, w, h, w * h, 0, x_end=x_end, y_end=y_end, cols=cols, rows=rows)
            else:
                x, y = m.new_int_var(0, d, f"{r.id}_x"), m.new_int_var(0, d, f"{r.id}_y")
                ws = [s[0] for s in r.shapes]
                hs = [s[1] for s in r.shapes]
                areas = [w * h for w, h in r.shapes]
                w = m.new_int_var(min(ws), max(ws), f"{r.id}_w")
                h = m.new_int_var(min(hs), max(hs), f"{r.id}_h")
                area = m.new_int_var(min(areas), max(areas), f"{r.id}_area")
                devs = [abs(a - r.target) for a in areas]
                dev = m.new_int_var(0, max(devs), f"{r.id}_dev")
                m.add_allowed_assignments([w, h, area, dev], [(sw, sh, a, e) for (sw, sh), a, e in
                                                             zip(r.shapes, areas, devs)])
                x_end, y_end = m.new_int_var(0, d, f"{r.id}_xe"), m.new_int_var(0, d, f"{r.id}_ye")
                m.add(x_end == x + w + C.INTERIOR_WALL)
                m.add(y_end == y + h + C.INTERIOR_WALL)
                rv = RoomVars(x, y, w, h, area, dev, x_end=x_end, y_end=y_end,
                              cols=self._span(x, x_end, self.xs, f"{r.id}_c"),
                              rows=self._span(y, y_end, self.ys, f"{r.id}_r"))
            rv.cell_x = m.new_interval_var(rv.x, rv.w + C.INTERIOR_WALL, rv.x_end, f"{r.id}_cx")
            rv.cell_y = m.new_interval_var(rv.y, rv.h + C.INTERIOR_WALL, rv.y_end, f"{r.id}_cy")
            self.rooms[r.id] = rv

    def _add_occupancy(self) -> None:
        m, n = self.m, self.n
        self.empty: dict[int, list[list]] = {}
        self.taken: dict[str, list[list]] = {}
        for floor in range(1, self.b.floors + 1):
            rooms = self.b.floor_rooms(floor)
            for r in rooms:
                self.taken[r.id] = [[m.new_bool_var(f"{r.id}_at{i}_{j}") for j in range(n)] for i in range(n)]
            m.add_no_overlap_2d([self.rooms[r.id].cell_x for r in rooms], [self.rooms[r.id].cell_y for r in rooms])
            empty = [[m.new_bool_var(f"f{floor}_empty{i}_{j}") for j in range(n)] for i in range(n)]
            for i in range(n):
                for j in range(n):
                    taken = []
                    for r in rooms:
                        rv, a = self.rooms[r.id], self.taken[r.id][i][j]
                        m.add_implication(a, rv.cols.cover[i])
                        m.add_implication(a, rv.rows.cover[j])
                        m.add_bool_or([~rv.cols.cover[i], ~rv.rows.cover[j], a])
                        taken.append(a)
                    m.add_exactly_one(taken + [empty[i][j]])
            self.empty[floor] = empty
            if self.rectangular:
                self._add_frame(empty, floor)
            else:
                self._add_open_outline(empty, floor)
                self._add_gaps(empty, self.xs, lambda i, j: empty[i][j])
                self._add_gaps(empty, self.ys, lambda i, j: empty[j][i])

        bottom = self.empty[1]
        m.add_bool_or([~bottom[0][j] for j in range(n)])
        m.add_bool_or([~bottom[n - 1][j] for j in range(n)])
        m.add_bool_or([~bottom[i][0] for i in range(n)])
        m.add_bool_or([~bottom[i][n - 1] for i in range(n)])
        for floor in range(2, self.b.floors + 1):
            for i in range(n):
                for j in range(n):
                    m.add_implication(self.empty[floor - 1][i][j], self.empty[floor][i][j])

    def _add_frame(self, empty: list[list], floor: int) -> None:
        """Прямоугольный контур этажа: занятые клетки образуют прямоугольник сетки."""
        m, n, d = self.m, self.n, self.size
        x0, y0, x1, y1 = (m.new_int_var(0, d, f"frame{floor}_{k}") for k in ("x0", "y0", "x1", "y1"))
        cols = self._span(x0, x1, self.xs, f"frame{floor}_c")
        rows = self._span(y0, y1, self.ys, f"frame{floor}_r")
        if floor == 1:
            m.add(cols.first[0] == 1)
            m.add(rows.first[0] == 1)
            m.add(cols.last[n - 1] == 1)
            m.add(rows.last[n - 1] == 1)
        for i in range(n):
            for j in range(n):
                m.add_bool_or([~cols.cover[i], ~rows.cover[j], ~empty[i][j]])
                m.add_implication(~empty[i][j], cols.cover[i])
                m.add_implication(~empty[i][j], rows.cover[j])
        self.frames[floor] = (x0, y0, x1, y1)
        self.frame_spans[floor] = (cols, rows)

    def _add_open_outline(self, empty: list[list], floor: int) -> None:
        """Каждая пустая клетка видит край сетки по прямой через пустые клетки."""
        m, n = self.m, self.n
        rays = {}
        for di, dj, tag in ((-1, 0, "w"), (1, 0, "e"), (0, -1, "s"), (0, 1, "n")):
            order_i = range(n) if di <= 0 else range(n - 1, -1, -1)
            order_j = range(n) if dj <= 0 else range(n - 1, -1, -1)
            for i in order_i:
                for j in order_j:
                    ray = m.new_bool_var(f"f{floor}_ray{tag}{i}_{j}")
                    m.add_implication(ray, empty[i][j])
                    pi, pj = i + di, j + dj
                    if 0 <= pi < n and 0 <= pj < n:
                        m.add_implication(ray, rays[(pi, pj, tag)])
                    rays[(i, j, tag)] = ray
        for i in range(n):
            for j in range(n):
                m.add_bool_or([rays[(i, j, t)] for t in "wesn"]).only_enforce_if(empty[i][j])

    def _add_gaps(self, empty, lines: list, cell) -> None:
        """Пустой промежуток между двумя занятыми клетками одного ряда не уже GAP_MIN."""
        m, n = self.m, self.n
        for j in range(n):
            for a in range(n):
                for b in range(a + 2, n):
                    lits = [~cell(a, j)] + [cell(k, j) for k in range(a + 1, b)] + [~cell(b, j)]
                    m.add(lines[b] - lines[a + 1] >= GAP_MIN).only_enforce_if(lits)

    def _share_wall(self, a: RoomVars, c: RoomVars, span: int, name: str) -> list:
        """Булевы «a западнее, восточнее, южнее, севернее c» с общим участком не короче span."""
        m = self.m
        rel = [m.new_bool_var(f"{name}_{side}") for side in ("w", "e", "s", "n")]
        m.add(a.x_end == c.x).only_enforce_if(rel[0])
        m.add(c.x_end == a.x).only_enforce_if(rel[1])
        m.add(a.y_end == c.y).only_enforce_if(rel[2])
        m.add(c.y_end == a.y).only_enforce_if(rel[3])
        for lit in rel[:2]:
            m.add(a.y + span <= c.y + c.h).only_enforce_if(lit)
            m.add(c.y + span <= a.y + a.h).only_enforce_if(lit)
        for lit in rel[2:]:
            m.add(a.x + span <= c.x + c.w).only_enforce_if(lit)
            m.add(c.x + span <= a.x + a.w).only_enforce_if(lit)
        # Та же связь на сетке: общая линия и хотя бы один общий ряд клеток.
        for lit, (lo, hi), across in ((rel[0], (a.cols, c.cols), (a.rows, c.rows)),
                                      (rel[1], (c.cols, a.cols), (a.rows, c.rows)),
                                      (rel[2], (a.rows, c.rows), (a.cols, c.cols)),
                                      (rel[3], (c.rows, a.rows), (a.cols, c.cols))):
            m.add_implication(lit, ~lo.last[-1])
            m.add_implication(lit, ~hi.first[0])
            for k in range(self.n - 1):
                m.add(lo.last[k] == hi.first[k + 1]).only_enforce_if(lit)
            common = []
            for j in range(self.n):
                both = m.new_bool_var("")
                m.add_implication(both, across[0].cover[j])
                m.add_implication(both, across[1].cover[j])
                common.append(both)
            m.add_bool_or(common).only_enforce_if(lit)
        return rel

    def _add_doors(self) -> None:
        for a, c in self.b.doors:
            rel = self._share_wall(self.rooms[a], self.rooms[c], DOOR_SPAN, f"door_{a}_{c}")
            self.m.add_exactly_one(rel)

    def _add_floor_connectivity(self) -> None:
        """Если двери этажа не связывают его комнаты, компоненты обязаны соприкасаться стенами."""
        for floor in range(1, self.b.floors + 1):
            ids = [r.id for r in self.b.floor_rooms(floor)]
            adjacency = {i: set() for i in ids}
            for a, c in self.b.doors:
                if a in adjacency:
                    adjacency[a].add(c)
                    adjacency[c].add(a)
            components, seen = [], set()
            for start in ids:
                if start in seen:
                    continue
                comp, queue = [start], deque([start])
                seen.add(start)
                while queue:
                    for nb in adjacency[queue.popleft()]:
                        if nb not in seen:
                            seen.add(nb)
                            comp.append(nb)
                            queue.append(nb)
                components.append(comp)
            for i in range(1, len(components)):
                touches = []
                for a in components[i]:
                    for j in range(i):
                        for c in components[j]:
                            touches += self._share_wall(self.rooms[a], self.rooms[c], 1, f"touch_{a}_{c}")
                self.m.add_bool_or(touches)

    def _add_openings(self) -> None:
        """Проём лежит на внешнем участке.

        Полоса в один шаг сразу за ячейкой комнаты вдоль проёма, с запасом на наружную стену
        соседа во внутреннем углу, не пересекается ни с одной ячейкой этажа.
        """
        m, d = self.m, self.size
        for r in self.b.rooms:
            rv = self.rooms[r.id]
            floor = [self.rooms[s.id] for s in self.b.floor_rooms(r.floor)]
            per_facade: dict[str, list] = {f: [] for f in C.FACADES}
            previous: dict[tuple, OpeningVars] = {}
            for k, o in enumerate(openings_of(r)):
                name = f"{r.id}_{o.kind}{k}"
                start = m.new_int_var(0, d, f"{name}_at")
                end = start + o.width
                facades = [o.facade] if o.facade else list(C.FACADES)
                on_facade = {}
                for facade in facades:
                    on = m.new_bool_var(f"{name}_{facade}")
                    face_lo, face_hi = (rv.x, rv.x + rv.w) if facade in HORIZONTAL else (rv.y, rv.y + rv.h)
                    m.add(start >= face_lo + C.OPENING_END_GAP).only_enforce_if(on)
                    m.add(end <= face_hi - C.OPENING_END_GAP).only_enforce_if(on)
                    per_facade[facade].append(
                        m.new_optional_fixed_size_interval_var(start, o.width + C.OPENING_GAP, on, f"{name}_{facade}_iv"))
                    on_facade[facade] = on
                    if self.rectangular:
                        m.add(self._face(rv, facade) == self._frame_side(r.floor, facade)).only_enforce_if(on)
                        mine, frame = self._edge_lines(rv.cols, rv.rows, facade), self._edge_lines(
                            *self.frame_spans[r.floor], facade)
                        for a, c in zip(mine, frame):
                            m.add(a == c).only_enforce_if(on)
                        continue
                    along = m.new_optional_fixed_size_interval_var(
                        start - NEAR_CLEARANCE, o.width + NEAR_CLEARANCE + FAR_CLEARANCE, on, f"{name}_{facade}_along")
                    band = m.new_optional_fixed_size_interval_var(self._band(rv, facade), 1, on, f"{name}_{facade}_band")
                    probe = (along, band) if facade in HORIZONTAL else (band, along)
                    m.add_no_overlap_2d([probe[0]] + [s.cell_x for s in floor], [probe[1]] + [s.cell_y for s in floor])
                    m.add_implication(on, self._facade_access(r, facade))
                m.add_exactly_one(on_facade.values())
                ov = OpeningVars(o, start, on_facade)
                key = (o.kind, o.width, o.facade)
                if o.facade and key in previous:
                    m.add(previous[key].start + o.width + C.OPENING_GAP <= start)
                previous[key] = ov
                self.openings.append(ov)
            for intervals in per_facade.values():
                if len(intervals) > 1:
                    m.add_no_overlap(intervals)

    def _facade_access(self, r: Room, facade: str):
        """Булева «у комнаты есть клетка сетки, за которой со стороны фасада пусто или край сетки»."""
        key = (r.id, facade)
        if key in self._access:
            return self._access[key]
        m, n, rv = self.m, self.n, self.rooms[r.id]
        empty, taken = self.empty[r.floor], self.taken[r.id]
        step = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}[facade]
        edge = {"north": rv.rows.last, "south": rv.rows.first, "east": rv.cols.last, "west": rv.cols.first}[facade]
        lits = []
        for i in range(n):
            for j in range(n):
                lit = m.new_bool_var(f"{r.id}_{facade}_out{i}_{j}")
                m.add_implication(lit, taken[i][j])
                m.add_implication(lit, edge[j if facade in HORIZONTAL else i])
                oi, oj = i + step[0], j + step[1]
                if 0 <= oi < n and 0 <= oj < n:
                    m.add_implication(lit, empty[oi][oj])
                lits.append(lit)
        access = m.new_bool_var(f"{r.id}_{facade}_access")
        m.add_bool_or(lits).only_enforce_if(access)
        self._access[key] = access
        return access

    @staticmethod
    def _face(rv: RoomVars, facade: str):
        """Граница ячейки комнаты со стороны фасада."""
        return {"north": rv.y_end, "south": rv.y, "east": rv.x_end, "west": rv.x}[facade]

    @staticmethod
    def _edge_lines(cols: Span, rows: Span, facade: str) -> list:
        """Булевы «граница со стороны фасада проходит по линии k» в порядке k."""
        return {"north": rows.last, "south": rows.first, "east": cols.last, "west": cols.first}[facade]

    def _frame_side(self, floor: int, facade: str):
        x0, y0, x1, y1 = self.frames[floor]
        return {"north": y1, "south": y0, "east": x1, "west": x0}[facade]

    @staticmethod
    def _band(rv: RoomVars, facade: str):
        """Начало ряда клеток сразу за ячейкой комнаты со стороны фасада."""
        return {"north": rv.y_end, "south": rv.y - 1, "east": rv.x_end, "west": rv.x - 1}[facade]

    def _add_house_area(self) -> None:
        total = sum(rv.area for rv in self.rooms.values())
        target = self.b.house_area
        lo = math.ceil(target * (100 - C.AREA_TOLERANCE_PCT) / 100)
        hi = target * (100 + C.AREA_TOLERANCE_PCT) // 100
        self.m.add_linear_constraint(total, lo, hi)

    def _add_objectives(self) -> None:
        m, d = self.m, self.size
        grow = 2 * C.EXTERIOR_WALL - C.INTERIOR_WALL
        outer_w = m.new_int_var(1, d + grow, "outer_w")
        outer_h = m.new_int_var(1, d + grow, "outer_h")
        m.add(outer_w == self.xs[-1] + grow)
        m.add(outer_h == self.ys[-1] + grow)
        outline = m.new_int_var(1, (d + grow) ** 2, "outline_area")
        m.add_multiplication_equality(outline, [outer_w, outer_h])
        self.tight = m.new_bool_var("tight")
        for rid, rv in self.rooms.items():
            r = self.b.room(rid)
            if r.kind != "stair":
                m.add(rv.dev <= r.target * TIGHT_PCT // 100).only_enforce_if(self.tight)
        self.objectives = {
            "deviation": [rv.dev for rv in self.rooms.values()],
            "outline": [outline],
            "walls": [v for rv in self.rooms.values() for v in (rv.w, rv.h)] + [self.xs[-1], self.ys[-1]],
        }

    def find_feasible(self, budget: float, tight: bool = False) -> tuple[int, list[int] | None]:
        """Первое допустимое решение без цели: с целью поиск сразу тянется к точным площадям и вязнет."""
        domain = self.m.proto.variables[self.tight.index].domain
        domain.clear()
        domain.extend([int(tight), int(tight)])
        solver = self._solver(budget)
        status = solver.solve(self.m)
        domain.clear()
        domain.extend([0, 1])
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return cp_model.FEASIBLE, list(solver.response_proto.solution)
        if status == cp_model.MODEL_INVALID:
            raise RuntimeError(f"модель CP-SAT некорректна: {self.m.validate()}")
        return status, None

    def optimize(self, solution: list[int], deadline: float) -> Plan:
        """Лексикографическая цель раздела 8: каждая стадия фиксирует достигнутое значение предыдущей."""
        shares = {"deviation": 0.75, "outline": 0.6, "walls": 1.0}
        values, statuses = {}, {}
        for name, terms in self.objectives.items():
            remaining = deadline - time.monotonic()
            if remaining > 0.05:
                self.m.minimize(sum(terms))
                self._hint(solution)
                solver = self._solver(shares[name] * remaining)
                status = solver.solve(self.m)
                statuses[name] = solver.status_name(status)
                if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    solution = list(solver.response_proto.solution)
            values[name] = sum(t if isinstance(t, int) else solution[t.index] for t in terms)
            self.m.add(sum(terms) <= values[name])
        plan = self._extract(solution)
        plan.stats = {"objective": values, "status": statuses}
        return plan

    def _hint(self, solution: list[int]) -> None:
        self.m.clear_hints()
        for i, value in enumerate(solution):
            self.m.add_hint(self.m.get_int_var_from_proto_index(i), value)

    @staticmethod
    def _solver(budget: float) -> cp_model.CpSolver:
        solver = cp_model.CpSolver()
        p = solver.parameters
        p.num_workers = 1
        p.random_seed = C.RANDOM_SEED
        p.max_time_in_seconds = max(budget, 0.01)
        p.linearization_level = 0
        return solver

    def _extract(self, solution: list[int]) -> Plan:
        def val(v):
            return v if isinstance(v, int) else solution[v.index]

        rooms = {rid: (val(rv.x), val(rv.y), val(rv.w), val(rv.h)) for rid, rv in self.rooms.items()}
        placements = []
        for ov in self.openings:
            facade = next(f for f, lit in ov.on_facade.items() if val(lit))
            placements.append(Placement(ov.opening, facade, val(ov.start)))
        return Plan(rooms, placements)


def solve(building: Building, time_limit: float = C.TIME_LIMIT_S) -> Plan:
    """Сначала прямоугольные этажи, затем общий контур с выемками на оставшееся время."""
    for r in building.rooms:
        if r.kind == "stair" and not any(r.id in pair for pair in building.doors):
            raise Infeasible(f"у лестничной комнаты {r.id} нет ребра в doors, а дверь на этаже шахты обязательна")
    started = time.monotonic()
    deadline = started + time_limit
    proven = []
    for rectangular, share in ((True, 0.5), (False, 1.0)):
        model = PlanModel(building, rectangular)
        mode_deadline = time.monotonic() + share * (deadline - time.monotonic())
        status, solution = model.find_feasible(TIGHT_SHARE * (mode_deadline - time.monotonic()), tight=True)
        if solution is None:
            status, solution = model.find_feasible(mode_deadline - time.monotonic())
        if solution is not None:
            plan = model.optimize(solution, deadline)
            plan.stats.update(outline="rectangular" if rectangular else "free",
                              seconds=round(time.monotonic() - started, 2))
            return plan
        proven.append(status == cp_model.INFEASIBLE)
    if all(proven):
        raise Infeasible(
            "ограничения планировки невыполнимы: допустимого плана нет ни с прямоугольными этажами, "
            f"ни с выемками контура (сетка до {GRID_MAX} линий, выемки не уже {GAP_MIN * C.UNIT_MM} мм)")
    raise Timeout(f"за {time_limit:g} с допустимый план не найден")
