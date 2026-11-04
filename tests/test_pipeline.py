import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from building_topology import constants as C
from building_topology.check import check
from building_topology.cli import main
from building_topology.layout import _spread, build, to_json
from building_topology.model import solve
from building_topology.problem import load, parse
from building_topology.render import render

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run_cli(*args: str) -> tuple[int, str]:
    err = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(err):
        code = main(list(args))
    return code, err.getvalue()


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

    def test_timeout(self):
        self.stale()
        code, err = run_cli(str(EXAMPLES / "07_large_2f.json"), "-o", str(self.out), "--time-limit", "0.2")
        self.assertEqual(code, 4)
        self.assertIn("лимит времени", err)
        self.assert_no_results()


if __name__ == "__main__":
    unittest.main()
