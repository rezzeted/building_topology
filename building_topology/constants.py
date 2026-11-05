"""Константы раздела 2 ТЗ. Длины — в шагах сетки, площади — в клетках сетки."""

UNIT_MM = 100
CELL_M2 = 0.01

INTERIOR_WALL = 1
EXTERIOR_WALL = 3
BEARING_WALL = 3

COLUMN = 3
BEAM_SPAN_DEFAULT = 60
BEAM_SPAN_MIN = 10

AREA_TOLERANCE_PCT = 10
ROOM_TOLERANCE_MAX_PCT = 50

MIN_SIDE = {"room": 16, "corridor": 12}
MAX_ASPECT = {"room": 3, "corridor": 8}
STAIR_MIN_SIDE = 20

INTERIOR_DOOR = 9
EXTERIOR_DOOR_DEFAULT = 10
EXTERIOR_DOOR_MIN = 9
WINDOW_DEFAULT = 12
WINDOW_MIN = 6

OPENING_END_GAP = 1
OPENING_GAP = 1

MAX_FLOORS = 5
MAX_ROOMS = 40

PX_PER_M = 100

TIME_LIMIT_S = 30.0
RANDOM_SEED = 1

FACADES = ("north", "east", "south", "west")
KINDS = ("room", "corridor", "stair")


def within_tolerance(actual: int, target: int, pct: int = AREA_TOLERANCE_PCT) -> bool:
    return 100 * abs(actual - target) <= pct * target
