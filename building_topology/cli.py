"""Запуск: python -m building_topology вход.json -o каталог [--time-limit секунды]."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from . import constants as C
from .check import check
from .layout import build, to_json
from .model import Infeasible, Timeout, solve
from .problem import InputError, load
from .render import render

OK, INPUT_ERROR, INFEASIBLE, TIMEOUT, INTERNAL = 0, 2, 3, 4, 1
RESULT = re.compile(r"layout\.json|floor-\d+\.(svg|png)")


def _positive(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("лимит времени должен быть больше нуля")
    return value


def _remove_results(out: Path) -> None:
    if out.is_dir():
        for path in out.iterdir():
            if RESULT.fullmatch(path.name):
                path.unlink()


def _fail(out: Path, code: int, message: str) -> int:
    _remove_results(out)
    print(message, file=sys.stderr)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m building_topology",
                                     description="Генератор планировки по графу здания (TZ.md).")
    parser.add_argument("input", help="входной JSON")
    parser.add_argument("-o", "--output", required=True, help="каталог результата")
    parser.add_argument("--time-limit", type=_positive, default=C.TIME_LIMIT_S,
                        help=f"лимит поиска в секундах, по умолчанию {C.TIME_LIMIT_S:g}")
    args = parser.parse_args(argv)
    out = Path(args.output)

    try:
        building = load(args.input)
    except InputError as exc:
        return _fail(out, INPUT_ERROR, f"ошибка входа: {exc}")
    try:
        plan = solve(building, args.time_limit)
    except Infeasible as exc:
        return _fail(out, INFEASIBLE, f"план невозможен: {exc}")
    except Timeout as exc:
        return _fail(out, TIMEOUT, f"лимит времени: {exc}")

    layout = build(building, plan)
    errors = check(layout)
    if errors:
        return _fail(out, INTERNAL, "внутренняя ошибка: план не прошёл проверку:\n  " + "\n  ".join(errors))

    data = to_json(layout)
    sheets = render(layout)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out, prefix=".build-") as tmp:
        staged = Path(tmp)
        (staged / "layout.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for floor, (svg, png) in sheets.items():
            (staged / f"floor-{floor}.svg").write_text(svg, encoding="utf-8")
            (staged / f"floor-{floor}.png").write_bytes(png)
        _remove_results(out)
        for path in staged.iterdir():
            shutil.move(str(path), out / path.name)

    m = data["metrics"]
    beams = f"балки {m['beam_length_m']:.1f} м, колонн {m['columns']}, " if m["beam_length_m"] or m["columns"] else ""
    off = m["rooms_out_of_tolerance"]
    off = f"комнат дальше {C.AREA_TOLERANCE_PCT}% от цели {off}, " if off else ""
    print(f"план: этажей {building.floors}, комнат {len(building.rooms)}, "
          f"отклонение площадей {m['area_deviation_m2']:.2f} м², {off}"
          f"контур этажа 1 {m['outline_area_m2']:.2f} м², стены {m['wall_length_m']:.1f} м, "
          f"{beams}{m['seconds']:.1f} с -> {out}")
    return OK
