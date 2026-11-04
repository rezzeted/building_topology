import json
import unittest
from pathlib import Path

from building_topology.problem import InputError, parse, rectangles

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def tz_example() -> dict:
    return json.loads((EXAMPLES / "05_tz_example_2f.json").read_text(encoding="utf-8"))


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


class InvalidInputTest(unittest.TestCase):
    def assert_rejected(self, data: dict, fragment: str):
        with self.assertRaises(InputError) as ctx:
            parse(data)
        self.assertIn(fragment, str(ctx.exception))

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


if __name__ == "__main__":
    unittest.main()
