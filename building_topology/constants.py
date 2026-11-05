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

# Отопитель в плане: вдоль стены, к которой он стоит спиной, и вглубь комнаты.
HEATER_SIZE = {"fireplace": (10, 6), "solid_boiler": (6, 12), "gas_boiler": (5, 4), "gas_convector": (6, 3)}
HEATER_KINDS = tuple(HEATER_SIZE)
GAS_KINDS = ("gas_boiler", "gas_convector")
FLUES = {"fireplace": ("chimney", "facade"), "solid_boiler": ("chimney",), "gas_boiler": ("wall",),
         "gas_convector": ("wall",)}
# Дымовая труба вдоль стены и вглубь: канал 140×270 мм в кирпичных стенках 120 мм, 510×380 мм.
CHIMNEY = (6, 4)
# Свободная зона перед отопителем: 1250 мм от топки до стены по сетке, проход перед конвектором.
FRONT_ZONE = {"fireplace": 13, "solid_boiler": 13, "gas_convector": 10}
CONVECTOR_SIDE_GAP = 2
CONVECTOR_ROOM_MIN = 600
CONVECTOR_PAIR_AREA = 2000
CONVECTOR_MAX_FLOORS = 2
BOILERS_PER_ROOM = 2
COAXIAL_MAX_FLOORS = 3
OUTLET_TO_OPENING = 5
OUTLET_GAP = 10
CONVECTOR_OUTLET_GAP = 3
# Зона двери, в которую не заходят отопители и трубы: по обе стороны стены и внутрь при двери наружу.
DOOR_ZONE = 9
DOOR_APPROACH = 5

MAX_FLOORS = 5
MAX_ROOMS = 40

PX_PER_M = 100

TIME_LIMIT_S = 420.0
RANDOM_SEED = 1

FACADES = ("north", "east", "south", "west")
KINDS = ("room", "corridor", "stair")
USES = ("living", "bedroom", "office", "kitchen", "dining", "hall", "vestibule", "bath", "wc", "storage",
        "boiler_room", "garage", "other")


def within_tolerance(actual: int, target: int, pct: int = AREA_TOLERANCE_PCT) -> bool:
    return 100 * abs(actual - target) <= pct * target
