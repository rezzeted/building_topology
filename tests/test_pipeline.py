import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from building_topology import constants as C
from building_topology.check import check
from building_topology.cli import main
from building_topology.layout import Beam, Column, Layout, Rect, _spread, area_flag, build, structure, to_json
from building_topology.model import solve
from building_topology.problem import load, parse
from building_topology.render import FILL, render

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run_cli(*args: str) -> tuple[int, str]:
    err = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(err):
        code = main(list(args))
    return code, err.getvalue()


def tower(**structure) -> dict:
    """Два этажа: внизу прихожая и лестница, наверху спальня меньше прихожей и та же лестница."""
    def room(rid, kind, floor, area, **extra):
        return {"id": rid, "name": rid, "kind": kind, "floor": floor, "area_m2": area, "windows": [], **extra}

    data = {
        "version": 1, "floors": 2, "house_area_m2": 33.52,
        "rooms": [room("hall", "corridor", 1, 14.0, exterior_doors=[{"facade": "east", "entrance": True}]),
                  room("stair1", "stair", 1, 5.76), room("bed", "room", 2, 8.0), room("stair2", "stair", 2, 5.76)],
        "doors": [{"a": "hall", "b": "stair1"}, {"a": "bed", "b": "stair2"}],
        "stairs": [{"id": "s", "from_floor": 1, "to_floor": 2, "width_mm": 2400, "depth_mm": 2400,
                    "rooms": ["stair1", "stair2"]}],
    }
    if structure:
        data["structure"] = structure
    return data


def closet_house() -> dict:
    """Кладовая 2 м²: самая малая комната 1600×1600 мм больше цели на 28%, это в допуске 30%."""
    return {
        "version": 1, "floors": 1, "house_area_m2": 12.0, "room_area_tolerance_pct": 30,
        "rooms": [{"id": "hall", "name": "Прихожая", "kind": "corridor", "floor": 1, "area_m2": 10.0, "windows": [],
                   "exterior_doors": [{"facade": "south", "entrance": True}]},
                  {"id": "closet", "name": "Кладовая", "kind": "room", "floor": 1, "area_m2": 2.0, "windows": []}],
        "doors": [{"a": "hall", "b": "closet"}],
    }


class SpreadTest(unittest.TestCase):
    def test_openings_keep_gaps(self):
        for widths, lo, hi in (([12], 0, 40), ([12, 12, 10], 3, 50), ([6, 6], 0, 13)):
            starts = _spread(widths, lo, hi)
            self.assertGreaterEqual(starts[0], lo)
            self.assertLessEqual(starts[-1] + widths[-1], hi)
            for (a, wa), b in zip(zip(starts, widths), starts[1:]):
                self.assertGreaterEqual(b - a - wa, C.OPENING_GAP)


class PipelineTest(unittest.TestCase):
    def test_tz_example(self):
        b = load(EXAMPLES / "05_tz_example_2f.json")
        layout = build(b, solve(b, 20))
        self.assertEqual(check(layout), [])
        data = to_json(layout)
        self.assertEqual({r["id"] for r in data["rooms"]}, {r.id for r in b.rooms})
        outer = [w for w in data["walls"] if w["kind"] == "exterior"]
        self.assertEqual(min(min(w["start"][0], w["end"][0]) for w in outer), C.EXTERIOR_WALL * C.UNIT_MM // 2)
        stairs = [r for r in data["rooms"] if r["id"].startswith("stair")]
        self.assertEqual(len({(r["x"], r["y"], r["w"], r["h"]) for r in stairs}), 1)
        doors = [o for o in data["openings"] if o["kind"] == "door"]
        self.assertEqual(sum(1 for o in doors if o.get("entrance")), 1)
        self.assertEqual(len(doors), len(b.doors) + 1)
        sheets = render(layout)
        self.assertEqual(sorted(sheets), [1, 2])
        svg, png = sheets[1]
        self.assertIn("1:100", svg)
        self.assertTrue(png.startswith(b"\x89PNG"))


class AreaFlagTest(unittest.TestCase):
    def test_flag_by_normal_tolerance(self):
        b = parse(closet_house())
        closet = b.room("closet")
        self.assertEqual([area_flag(closet, a) for a in (170, 180, 220, 230)], ["smaller", None, None, "larger"])
        stair = parse(tower()).room("stair1")
        self.assertIsNone(area_flag(stair, 2 * stair.target))

    def test_marked_in_json_and_drawing(self):
        b = parse(closet_house())
        layout = build(b, solve(b, 10))
        self.assertEqual(check(layout), [])
        data = to_json(layout)
        rooms = {r["id"]: r for r in data["rooms"]}
        self.assertEqual(rooms["closet"]["area_flag"], "larger")
        self.assertEqual(rooms["closet"]["target_area_m2"], 2.0)
        self.assertNotIn("area_flag", rooms["hall"])
        self.assertEqual(data["metrics"]["rooms_out_of_tolerance"], 1)
        svg, _ = render(layout)[1]
        self.assertIn(f'fill="{FILL["larger"]}"', svg)
        self.assertIn("Зелёная заливка — Кладовая", svg)


class StructureTest(unittest.TestCase):
    """Нижний этаж — комната 6×4 м, верхний — левая половина: восточная стена верха висит над комнатой."""

    def layout(self, span_mm: int) -> Layout:
        b = parse(tower(max_beam_span_mm=span_mm))
        rooms = {"low": Rect("low", 1, 0, 0, 60, 40), "up": Rect("up", 2, 0, 0, 30, 40)}
        return Layout(b, rooms, {}, [], [])

    def test_beam_under_hanging_wall(self):
        beams, columns = structure(self.layout(6000))
        self.assertEqual([frozenset((b.start, b.end)) for b in beams], [frozenset(((31, -1), (31, 40)))])
        self.assertEqual(beams[0].floor, 1)
        self.assertEqual(columns, [])

    def test_long_span_gets_column(self):
        beams, columns = structure(self.layout(3000))
        self.assertEqual(len(beams), 2)
        self.assertTrue(all(b.length <= 30 for b in beams))
        self.assertEqual(columns, [Column(1, (31, 19))])


class SupportTest(unittest.TestCase):
    def test_beams_found_and_checked(self):
        b = parse(tower(allow_beams=True))
        layout = build(b, solve(b, 10))
        self.assertEqual(check(layout), [])
        self.assertTrue(layout.beams)
        data = to_json(layout)
        self.assertEqual(len(data["beams"]), len(layout.beams))
        self.assertGreater(data["metrics"]["beam_length_m"], 0)
        layout.beams, layout.columns = [], []
        self.assertTrue(any("без опоры" in e for e in check(layout)))

    def test_without_beams_uses_bearing_walls(self):
        b = load(EXAMPLES / "05_tz_example_2f.json")
        self.assertFalse(b.allow_beams)
        layout = build(b, solve(b, 10))
        self.assertEqual(check(layout), [])
        self.assertEqual((layout.beams, layout.columns), ([], []))
        bearing = [w for w in to_json(layout)["walls"] if w["kind"] == "bearing"]
        self.assertTrue(bearing)
        self.assertTrue(all(w["thickness_mm"] == C.BEARING_WALL * C.UNIT_MM for w in bearing))
        layout.beams = [Beam(1, (0, 0), (0, 5))]
        self.assertIn("в плане есть балки или колонны, а structure.allow_beams не задан", check(layout))


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def stale(self):
        self.out.mkdir(parents=True, exist_ok=True)
        for name in ("layout.json", "floor-1.svg", "floor-1.png", "notes.txt"):
            (self.out / name).write_text("old")

    def assert_no_results(self):
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["notes.txt"])

    def test_success(self):
        self.stale()
        code, _ = run_cli(str(EXAMPLES / "01_studio_1f.json"), "-o", str(self.out), "--time-limit", "10")
        self.assertEqual(code, 0)
        names = sorted(p.name for p in self.out.iterdir())
        self.assertEqual(names, ["floor-1.png", "floor-1.svg", "layout.json", "notes.txt"])
        data = json.loads((self.out / "layout.json").read_text(encoding="utf-8"))
        self.assertEqual(data["version"], 1)

    def test_input_error(self):
        self.stale()
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text('{"version": 2}', encoding="utf-8")
        code, err = run_cli(str(bad), "-o", str(self.out))
        self.assertEqual(code, 2)
        self.assertIn("ошибка входа", err)
        self.assert_no_results()

    def test_infeasible(self):
        self.stale()
        data = json.loads((EXAMPLES / "05_tz_example_2f.json").read_text(encoding="utf-8"))
        data["rooms"] = [r for r in data["rooms"] if r["id"] not in ("hall2", "bed")]
        data["doors"] = [d for d in data["doors"] if "hall2" not in (d["a"], d["b"])]
        data["house_area_m2"] = sum(r["area_m2"] for r in data["rooms"])
        parse(data)
        path = Path(self.tmp.name) / "lonely_stair.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        code, err = run_cli(str(path), "-o", str(self.out))
        self.assertEqual(code, 3)
        self.assertIn("stair2", err)
        self.assert_no_results()

    def test_hanging_walls_without_beams(self):
        self.stale()
        path = Path(self.tmp.name) / "tower.json"
        path.write_text(json.dumps(tower()), encoding="utf-8")
        code, err = run_cli(str(path), "-o", str(self.out), "--time-limit", "10")
        self.assertEqual(code, 3)
        self.assertIn("allow_beams", err)
        self.assert_no_results()

    def test_timeout(self):
        self.stale()
        code, err = run_cli(str(EXAMPLES / "07_large_2f.json"), "-o", str(self.out), "--time-limit", "0.2")
        self.assertEqual(code, 4)
        self.assertIn("лимит времени", err)
        self.assert_no_results()


if __name__ == "__main__":
    unittest.main()
