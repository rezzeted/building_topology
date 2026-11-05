import json
import unittest
from pathlib import Path

from building_topology.problem import InputError, parse, rectangles

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def tz_example() -> dict:
    return json.loads((EXAMPLES / "05_tz_example_2f.json").read_text(encoding="utf-8"))


def heated(*heaters: dict, **uses: str) -> dict:
    """Пример ТЗ с отопителями и назначениями комнат."""
    data = tz_example()
    for r in data["rooms"]:
        if r["id"] in uses:
            r["use"] = uses[r["id"]]
    data["heaters"] = list(heaters)
    return data


def heater(hid: str, kind: str, room: str, **extra) -> dict:
    return {"id": hid, "kind": kind, "room": room, **extra}


class RejectsMixin:
    def assert_rejected(self, data: dict, fragment: str):
        with self.assertRaises(InputError) as ctx:
            parse(data)
        self.assertIn(fragment, str(ctx.exception))


class ExamplesTest(unittest.TestCase):
    def test_all_examples_are_valid(self):
        for path in sorted(EXAMPLES.glob("*.json")):
            with self.subTest(path.name):
                parse(json.loads(path.read_text(encoding="utf-8")))

    def test_stair_room_gets_shaft_shape(self):
        b = parse(tz_example())
        self.assertEqual(b.room("stair1").shapes, [(24, 24)])
        self.assertEqual(b.room("stair2").shaft, "stair_main")


class RectanglesTest(unittest.TestCase):
    def test_bounds(self):
        for w, h in rectangles("room", 1600):
            self.assertGreaterEqual(min(w, h), 16)
            self.assertLessEqual(max(w, h), 3 * min(w, h))
            self.assertLessEqual(abs(w * h - 1600) * 10, 1600)
        self.assertIn((40, 40), rectangles("room", 1600))
        self.assertEqual(rectangles("room", 100), [])


class InvalidInputTest(RejectsMixin, unittest.TestCase):
    def test_unknown_field(self):
        data = tz_example()
        data["rooms"][0]["color"] = "red"
        self.assert_rejected(data, "неизвестное поле")

    def test_length_not_on_grid(self):
        data = tz_example()
        data["rooms"][1]["windows"][0]["width_mm"] = 1250
        self.assert_rejected(data, "не кратно")

    def test_house_area_out_of_tolerance(self):
        data = tz_example()
        data["house_area_m2"] = 80
        self.assert_rejected(data, "house_area_m2")

    def test_two_entrances(self):
        data = tz_example()
        data["rooms"][1]["exterior_doors"] = [{"facade": "west", "entrance": True}]
        self.assert_rejected(data, "входов в дом 2")

    def test_door_between_floors(self):
        data = tz_example()
        data["doors"].append({"a": "living", "b": "bed"})
        self.assert_rejected(data, "на разных этажах")

    def test_unreachable_room(self):
        data = tz_example()
        data["doors"] = [d for d in data["doors"] if d != {"a": "hall2", "b": "bed"}]
        self.assert_rejected(data, "не достижима")

    def test_stair_area_must_match_shaft(self):
        data = tz_example()
        stair = next(r for r in data["rooms"] if r["id"] == "stair1")
        stair["area_m2"] = 6.0
        self.assert_rejected(data, "stair1")

    def test_room_without_rectangle(self):
        data = tz_example()
        data["rooms"][0]["area_m2"] = 1.0
        data["house_area_m2"] = sum(r["area_m2"] for r in data["rooms"])
        self.assert_rejected(data, "hall")

    def test_structure_fields(self):
        data = tz_example()
        data["structure"] = {"allow_beams": "yes"}
        self.assert_rejected(data, "allow_beams")
        data["structure"] = {"max_beam_span_mm": 900}
        self.assert_rejected(data, "меньше минимума")
        data["structure"] = {"strict_support": True}
        self.assert_rejected(data, "неизвестное поле")

    def test_room_tolerance_bounds(self):
        for value in (0, 51, 12.5, "20"):
            with self.subTest(value):
                data = tz_example()
                data["room_area_tolerance_pct"] = value
                self.assert_rejected(data, "room_area_tolerance_pct")


class StructureTest(unittest.TestCase):
    def test_defaults(self):
        b = parse(tz_example())
        self.assertFalse(b.allow_beams)
        self.assertEqual(b.beam_span, 60)

    def test_given(self):
        data = tz_example()
        data["structure"] = {"allow_beams": True, "max_beam_span_mm": 4500}
        b = parse(data)
        self.assertTrue(b.allow_beams)
        self.assertEqual(b.beam_span, 45)


class RoomToleranceTest(unittest.TestCase):
    def test_default(self):
        b = parse(tz_example())
        self.assertEqual(b.room_tolerance, 10)
        self.assertLessEqual(max(w * h for w, h in b.room("bed").shapes), 1540)

    def test_wider_tolerance_gives_more_shapes(self):
        data = tz_example()
        data["room_area_tolerance_pct"] = 25
        b = parse(data)
        self.assertEqual(b.room_tolerance, 25)
        areas = [w * h for w, h in b.room("bed").shapes]
        self.assertGreater(max(areas), 1540)
        self.assertLess(min(areas), 1260)
        self.assertTrue(all(abs(a - 1400) * 4 <= 1400 for a in areas))


class HeaterInputTest(RejectsMixin, unittest.TestCase):
    def test_defaults(self):
        b = parse(heated(heater("fp", "fireplace", "living"), heater("gb", "gas_boiler", "kitchen"), kitchen="kitchen"))
        self.assertEqual([(h.flue, h.floor) for h in b.heaters], [("chimney", 1), ("wall", 1)])
        self.assertEqual((b.room("living").use, b.room("kitchen").use), ("other", "kitchen"))
        self.assertEqual((b.gas_room("living"), b.gas_room("kitchen")), (False, True))

    def test_use_values(self):
        self.assert_rejected(heated(living="attic"), "use 'attic'")
        self.assert_rejected(heated(stair1="hall"), "у лестничной комнаты use не задаётся")

    def test_fields(self):
        self.assert_rejected(heated(heater("hall", "fireplace", "living")), "идентификатор повторяется")
        self.assert_rejected(heated(heater("h", "stove", "living")), "kind 'stove'")
        self.assert_rejected(heated(heater("h", "fireplace", "attic")), "нет комнаты 'attic'")
        self.assert_rejected(heated(heater("h", "fireplace", "living", flue="wall")), "flue 'wall'")
        self.assert_rejected(heated(heater("h", "fireplace", "stair1")), "в лестничной комнате")

    def test_boilers(self):
        self.assert_rejected(heated(heater("gb", "gas_boiler", "living")), "п. 5.5")
        self.assert_rejected(heated(heater("gb", "gas_boiler", "bed"), bed="kitchen"), "п. 7.13")
        self.assert_rejected(heated(heater("tb", "solid_boiler", "kitchen")), "boiler_room")
        three = [heater(f"gb{i}", "gas_boiler", "kitchen") for i in range(3)]
        self.assert_rejected(heated(*three, kitchen="kitchen"), "п. 5.7")
        data = heated(heater("gb", "gas_boiler", "kitchen"), kitchen="kitchen")
        data["rooms"][2]["windows"] = []
        self.assert_rejected(data, "п. 5.10")

    def test_convectors(self):
        self.assert_rejected(heated(heater("c", "gas_convector", "kitchen"), kitchen="bath"), "п. Е.1")
        pair = [heater("c1", "gas_convector", "living"), heater("c2", "gas_convector", "living")]
        self.assert_rejected(heated(*pair), "п. Е.3")
        data = heated(heater("c", "gas_convector", "living"))
        data["rooms"][1]["area_m2"] = 24.0
        data["house_area_m2"] += 8.0
        self.assert_rejected(data, "п. Е.2")
        data["rooms"][1]["area_m2"] = 21.0
        data["house_area_m2"] -= 3.0
        shapes = parse(data).room("living").shapes
        self.assertTrue(shapes)
        self.assertTrue(all(w * h < 2000 for w, h in shapes))

    def test_tall_house(self):
        data = json.loads((EXAMPLES / "08_compact_3f.json").read_text(encoding="utf-8"))
        data["heaters"] = [heater("c", "gas_convector", "living")]
        self.assert_rejected(data, "не выше 2 этажей")

    def test_gas_rooms_door(self):
        data = heated(heater("gb", "gas_boiler", "kitchen"), heater("c", "gas_convector", "living"), kitchen="kitchen")
        data["doors"].append({"a": "living", "b": "kitchen"})
        self.assert_rejected(data, "п. 5.11")

    def test_heater_does_not_fit(self):
        data = heated(heater("fp", "fireplace", "kitchen"))
        data["rooms"][2]["area_m2"] = 2.6
        data["house_area_m2"] -= 7.4
        self.assert_rejected(data, "не помещается отопитель")


if __name__ == "__main__":
    unittest.main()
