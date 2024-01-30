import argparse
import os
import shutil
from dataclasses import dataclass
import re
import sys
from itertools import chain, product
from typing import Iterable, List, Optional, Generator, Tuple
from solid2 import P3, scad_inline
from solid2.core.object_base import OpenSCADObject
from pathlib import Path
import subprocess


@dataclass
class RenderTask:
    scad_object: OpenSCADObject
    position: P3
    filename: Path


@dataclass
class _RenderTaskArgs:
    scad_object: OpenSCADObject
    position: P3
    filename: Path
    openscad_bin: Path | None
    verbose: bool


def mod_p3(v: P3 | List[float],
           x: float | None = None,
           y: float | None = None,
           z: float | None = None,
           ax: float | None = None,
           ay: float | None = None,
           az: float | None = None,
           mx: float | None = None,
           my: float | None = None,
           mz: float | None = None,
           s: Tuple[float | None, float | None, float | None] | None = None,  # set
           a: Tuple[float | None, float | None, float | None] | None = None,  # add
           m: Tuple[float | None, float | None, float | None] | None = None,  # multiply
           ) -> P3:
    set_values = (x, y, z)
    add_values = (ax, ay, az)
    mul_values = (mx, my, mz)
    vl: List[float] = [vv for vv in v]
    for i in range(3):
        if set_values[i] is not None and s is not None and s[i] is not None:
            raise ValueError(f"Cant decide which value to set, check if you use the x,y,z argument with the s tuple "
                             f"argument, values are {set_values[i]} and {s[i]} for the {i}nth P3 element")
        if set_values[i] is not None:
            vl[i] = set_values[i]
        if s is not None and s[i] is not None:
            vl[i] = s[i]
        if add_values[i] is not None:
            vl[i] += add_values[i]
        if a is not None and a[i] is not None:
            vl[i] += a[i]
        if mul_values[i] is not None:
            vl[i] *= mul_values[i]
        if m is not None and m[i] is not None:
            vl[i] *= m[i]

    return vl[0], vl[1], vl[2]



def make_RenderTaskArgs(task: RenderTask, openscad_bin: Path | None, verbose: bool) -> _RenderTaskArgs:
    return _RenderTaskArgs(task.scad_object, task.position, task.filename, openscad_bin, verbose)


def modify_render_task(render_tasks: Iterable[RenderTask], offset: P3 = (0.0, 0.0, 0.0), name_suffix: str = "") -> \
        Generator[
            RenderTask, None, None]:
    for task in render_tasks:
        pos = task.position
        new_pos = (pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2])
        new_name = Path(f"{task.filename}{name_suffix}")
        yield RenderTask(task.scad_object, new_pos, Path(new_name))


def _render_to_file(task: _RenderTaskArgs) -> Path:
    scad_filename = task.filename.with_suffix(".scad").absolute().as_posix()
    task.scad_object.save_as_scad(scad_filename)
    if task.openscad_bin is not None:
        out_filenames = tuple(chain.from_iterable(
            product(("-o",), (task.filename.with_suffix(ext).absolute().as_posix() for ext in (".3mf", ".png")))))
        openscad_cli_args = [task.openscad_bin, *out_filenames, "--colorscheme", "BeforeDawn", scad_filename]

        try:
            subprocess.check_output(openscad_cli_args)
        except subprocess.CalledProcessError as ex:
            print(ex, file=sys.stderr)
    return task.filename.absolute()


def save_to_file(output_scad_basename: Path, openscad_bin: Path | None, render_tasks: Iterable[RenderTask],
                 all_filename: Path = Path("all"), include_filter_regex: re.Pattern[str] | None = None,
                 verbose: bool = False) -> None:
    if verbose:
        from multiprocessing.dummy import Pool
    else:
        from multiprocessing import Pool

    render_tasks_args: List[_RenderTaskArgs] = [make_RenderTaskArgs(t, openscad_bin, verbose) for t in render_tasks]
    if include_filter_regex is not None:
        render_tasks_args = [t for t in render_tasks_args if include_filter_regex.search(t.filename.as_posix())]

    if verbose:
        print(f"Will generate ", end="")
        for task in render_tasks_args:
            print(f"{task.filename} ", end="")
        print()

    all_obj: Optional[OpenSCADObject] = None
    for task_args in render_tasks_args:
        task_args.filename = output_scad_basename.joinpath(task_args.filename)
        if verbose:
            task_args.scad_object += scad_inline(f'echo("Writing {task_args.filename.as_posix()}");\n')
        if all_obj is None:
            all_obj = task_args.scad_object.translate(task_args.position)
        else:
            all_obj += task_args.scad_object.translate(task_args.position)

    scad_filename = output_scad_basename.joinpath(all_filename).with_suffix(".scad")
    if all_obj is not None:
        all_obj.save_as_scad(scad_filename.absolute().as_posix())

    with Pool() as pool:
        pool.map(_render_to_file, render_tasks_args)


def solid2_utils_cli(prog: str, description: str, default_output_path: Path):
    parser = argparse.ArgumentParser(prog=prog,
                                     description=description)
    parser.add_argument('--skip_rendering', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--openscad_bin', type=str)
    parser.add_argument('--include_filter_regex', type=str)
    parser.add_argument('--build_dir', type=str)

    args = parser.parse_args()

    output_path = default_output_path if args.build_dir is None else Path(args.build_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    openscad_bin: Path | None = Path(args.openscad_bin) if args.openscad_bin is not None else shutil.which("openscad")
    if openscad_bin is None and not args.skip_rendering:
        print("Didn't found openscad in PATH environment variable, skipping rendering 3mf/stl/png!")
        if Path("C:/Program Files/Openscad/openscad.exe").exists():
            openscad_bin = Path("C:/Program Files/Openscad/openscad.exe").absolute()
            print(f"Found openscad in default folder {openscad_bin.absolute()}")

    if args.skip_rendering:
        openscad_bin = None
    return args, output_path, openscad_bin
