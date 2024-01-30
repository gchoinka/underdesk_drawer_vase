import re
from math import sin, pi, cos, floor, sqrt, atan2
from os import path
from pathlib import Path
from typing import Tuple, List

from solid2 import cube, cylinder, P3, polyhedron, hull, intersection, scale, sphere, minkowski
from solid2.core.object_base import OpenSCADObject

from solid2_utils.utils import save_to_file, RenderTask, mod_p3
from src.solid2_utils.utils import solid2_utils_cli

unprintable_thickness = 0.01
preview_fix = 0.05

preview = True
preview_faces_factor = 3


# TODO: no bottom layer?
# TODO: stackable
# TODO: saw profile for the sides
# TODO: handle
# TODO: drawer mount
# TODO: full extension drawer sliders
# TODO: end stop on sliders
# TODO: branding
# TODO: remove constants
# TODO: taper end of sidewall
# TODO: document M221 S150


def rotate_point(top_p: P3, bottom_p: P3, angle: float) -> P3:
    # Calculate the midpoint
    mid_p = ((top_p[0] + bottom_p[0]) / 2, top_p[1], (top_p[2] + bottom_p[2]) / 2)

    # Translate the point to the origin
    x = top_p[0] - mid_p[0]
    y = top_p[1]  # y-coordinate remains the same
    z = top_p[2] - mid_p[2]

    # Perform the rotation
    new_x = x * cos(angle) - z * sin(angle)
    new_z = x * sin(angle) + z * cos(angle)

    # Translate back
    new_x += mid_p[0]
    new_z += mid_p[2]

    return new_x, y, new_z


def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return atan2(dy, dx)


def corrugated_cube(s: P3 | List[float] = (1., 1., 1.), offset: float = 0, ridges: float = 4, ridges_height: float = 5,
                    f_amp=lambda x: 1, f_z=lambda x: 0, f_coordinates=lambda coord, s, ridges_height: coord,
                    low_poly_mode: bool = True, const_thickness: bool = False, center: bool = False) -> OpenSCADObject:
    if low_poly_mode:
        number_of_faces: int = int(ridges * 2)
    else:
        number_of_faces: int = int(floor(s[0] * preview_faces_factor) if preview else floor(s[0] * 10))
    x: list[float] = [i * s[0] / number_of_faces for i in range(number_of_faces)] + [s[0]]
    frequency = ridges * 2 / s[0]
    coordinates_sin = []
    coordinates_cos = []
    for i in range(len(x)):
        sin_arg = offset + (pi / 2) * frequency * x[i]
        coordinates_sin.append(sin(sin_arg))
        coordinates_cos.append(cos(sin_arg))

    prev_p: Tuple[int, ...] = (0, 0, 0, 0)
    t = s[2] / 2
    b = -s[2] / 2
    coordinates = []
    faces = []
    for i in range(len(x)):
        z = ridges_height / 2 * (1. + coordinates_sin[i]) * f_amp(x[i]) + f_z(x[i])
        p = tuple(idx + len(coordinates) for idx in range(4))
        c = [[x[i], 0., z + t], [x[i], float(s[1]), z + t], [x[i], 0., z + b], [x[i], float(s[1]), z + b]]
        coordinates += f_coordinates(c, s, ridges_height)
        if i == 0:
            faces.append([p[2], p[3], p[1], p[0]])
        else:
            faces.append([prev_p[1], p[1], p[0], prev_p[0]])  # top
            faces.append([prev_p[2], p[2], p[3], prev_p[3]])  # bottom
            faces.append([prev_p[0], p[0], p[2], prev_p[2]])  # side
            faces.append([prev_p[3], p[3], p[1], prev_p[1]])  # side

        prev_p = p
    faces.append([prev_p[0], prev_p[1], prev_p[3], prev_p[2]])

    if const_thickness:  # TODO still WIP, see error when ridges_height==0
        for i in range(len(x)):
            ti, bi = (i * 4 + 0, i * 4 + 2)  # index: top, bottom
            coordinates[ti] = rotate_point(coordinates[ti], coordinates[bi], coordinates_cos[i])
            coordinates[bi] = rotate_point(coordinates[bi], coordinates[ti], coordinates_cos[i])

            ti, bi = (i * 4 + 1, i * 4 + 3)  # index: top, bottom
            coordinates[ti] = rotate_point(coordinates[ti], coordinates[bi], coordinates_cos[i])
            coordinates[bi] = rotate_point(coordinates[bi], coordinates[ti], coordinates_cos[i])

    end_position = (0., 0., s[2] / 2) if not center else (-s[0] / 2, -s[1] / 2, -ridges_height / 2)
    root = polyhedron(coordinates, faces)

    return root.translate(end_position)


def corrugated_cube_on_one_side(s: P3 | List[float] = (1., 1., 1.), offset: float = 0, ridges: float = 4,
                                ridges_height: float = 5, f_amp=lambda x: 1, f_z=lambda x: 0,
                                low_poly_mode: bool = True, center: bool = False) -> OpenSCADObject:
    def make_one_side_flat(c: List[List[float]], _s: P3 | List[float], _ridges_height: float) -> List[List[float]]:
        c[0][2] = s[2] / 2 + ridges_height
        c[1][2] = s[2] / 2 + ridges_height
        return c

    return corrugated_cube(s=s, offset=offset, ridges=ridges, ridges_height=ridges_height, f_amp=f_amp, f_z=f_z,
                           f_coordinates=make_one_side_flat, low_poly_mode=low_poly_mode, center=center)


def corrugated_cube_on_one_side_front(s: P3 | List[float] = (1., 1., 1.), offset: float = 0, ridges: float = 4,
                                      ridges_height: float = 5, f_amp=lambda x: 1, f_z=lambda x: 0,
                                      low_poly_mode: bool = True, center: bool = False) -> OpenSCADObject:
    def make_one_side_flat(c: List[List[float]], _s: P3 | List[float], _ridges_height: float) -> List[List[float]]:
        c[0][2] = s[2] / 2 + ridges_height
        c[1][2] = s[2] / 2 + ridges_height
        # c[2][1] = 1 + s[2] / 3  # TODO fix it
        #  c[1][2] += 1 - s[1] / 3

        return c

    return corrugated_cube(s=s, offset=offset, ridges=ridges, ridges_height=ridges_height, f_amp=f_amp, f_z=f_z,
                           f_coordinates=make_one_side_flat, low_poly_mode=low_poly_mode, center=center)


def corrugated_cube_front(s: P3 | List[float] = (1., 1., 1.), offset: float = 0, ridges: float = 4,
                          ridges_height: float = 5, f_amp=lambda x: 1, f_z=lambda x: 0, low_poly_mode: bool = True,
                          top_offset: float = 0, center: bool = False) -> OpenSCADObject:
    def make_one_side_flat(c: List[List[float]], _s: P3 | List[float], _ridges_height: float) -> List[List[float]]:
        c[0][2] = s[2] / 2 + ridges_height
        c[1][2] = s[2] / 2 + ridges_height

        c[0][1] += top_offset + s[1] / 3  # TODO fix it
        c[1][1] += top_offset - s[1] / 3
        return c

    return corrugated_cube(s, offset, ridges, ridges_height, f_amp, f_z, make_one_side_flat, low_poly_mode, center)


def hexagon(s: P3 | list[float], center=False) -> OpenSCADObject:
    obj = cylinder(r=1, h=s[2], _fn=16, center=center)
    obj = scale(mod_p3(s, z=1))(obj).rotate([0, 0, 0])
    return obj


def hexagons(s: P3 | List[float], edge_distance: tuple[float, float, float, float], border: float, rows: int,
             columns: int) -> OpenSCADObject:
    radius_x = (s[0] - edge_distance[0] - edge_distance[2] - border * (columns - 1)) / columns / 2
    radius_y = (s[1] - edge_distance[1] - edge_distance[3] - border * (rows - 1)) / rows / 2
    obj = cube([0, 0, 0])

    y_pos = edge_distance[1]
    for r in range(rows):
        x_pos = edge_distance[0]
        for c in range(columns):
            obj += hexagon([radius_x, radius_y, s[2]], center=True).translate(
                [x_pos + radius_x, y_pos + radius_y, s[2] / 2])
            x_pos += border + radius_x * 2
        y_pos += border + radius_y * 2
    return obj


def back_wall_round(s: P3 | List[float], corner_r: float = 7, center: bool = True) -> OpenSCADObject:
    c0 = cylinder(r=corner_r, h=s[2], center=True).translate([+(s[0] / 2 - corner_r), +(s[1] / 2 - corner_r), s[2] / 2])
    c1 = cylinder(r=corner_r, h=s[2], center=True).translate([-(s[0] / 2 - corner_r), +(s[1] / 2 - corner_r), s[2] / 2])
    c2 = cylinder(r=corner_r, h=s[2], center=True).translate([-(s[0] / 2 - corner_r), -(s[1] / 2 - corner_r), s[2] / 2])
    c3 = cylinder(r=corner_r, h=s[2], center=True).translate([+(s[0] / 2 - corner_r), -(s[1] / 2 - corner_r), s[2] / 2])
    root = hull()(c0, c1, c2, c3)
    if center:
        offset = [0, 0, -s[2] / 2]
    else:
        offset = [s[0] / 2, s[1] / 2, 0]
    return root.translate(offset)


def back_wall_round_sparse(s: P3 | List[float], corner_r: float = 7, inner_wall_width: float = 1.25,
                           outer_wall_with: float = 3, columns: int = 15, rows: int = 4,
                           center: bool = True) -> OpenSCADObject:
    outside = back_wall_round(s=s, corner_r=corner_r, center=True)
    inside = back_wall_round(s=mod_p3(s, ax=-outer_wall_with * 2, ay=-outer_wall_with * 2),
                             corner_r=corner_r - outer_wall_with)

    columns_space = s[0] / (columns - 1)
    row_space = s[1] / (rows - 1)

    positions: List[P3] = list()
    for i in range(rows - 1):
        y_pos = row_space * i
        positions.append((- s[0] / 2, y_pos - s[1] / 2, 0.))

    for i in range(columns - 1):
        x_pos = columns_space * i
        positions.append((x_pos - s[0] / 2, s[1] / 2, 0.))

    for i in range(rows - 1):
        y_pos = row_space * i
        positions.append((s[0] / 2, -y_pos + s[1] / 2, 0.))

    for i in range(columns - 1):
        x_pos = columns_space * i
        positions.append((-x_pos + s[0] / 2, -s[1] / 2, 0.))

    obj = cube([0, 0, 0])
    for i in range(len(positions) - 1):
        p1 = positions[i + 1]
        p2 = positions[len(positions) - 1 - i]
        obj += hull()(cylinder(r=inner_wall_width / 2, h=s[2], center=True).translate(p1),
                      cylinder(r=inner_wall_width / 2, h=s[2], center=True).translate(p2))
    obj += obj.translate([s[0] / 2, 0, 0]).mirrorX().translate([s[0] / 2, 0, 0])

    offset = (0, 0, 0) if center else mod_p3(s, mx=0.5, my=0.5, mz=0.5)

    return (outside - inside + intersection()(obj, outside)).translate(offset)


# https://www.calculator.net
def bevel_cube(s: P3 | List[float], bevel_a: Tuple[float, float, float, float] | List[float] = (pi, pi, pi, pi),
               center: bool = False) -> OpenSCADObject:
    c = s[2]
    b_a = pi / 2
    top_reduction = tuple(c * sin(a) / sin(pi - a - b_a) for a in bevel_a)
    cube_points = [(0, 0, 0),  # 0
                   (s[0], 0, 0),  # 1
                   (s[0], s[1], 0),  # 2
                   (0, s[1], 0),  # 3
                   (0 + top_reduction[1], 0 + top_reduction[0], s[2]),  # 4 top
                   (s[0] - top_reduction[2], 0 + top_reduction[0], s[2]),  # 5 top
                   (s[0] - top_reduction[2], s[1] - top_reduction[3], s[2]),  # 6 top
                   (0 + top_reduction[1], s[1] - top_reduction[3], s[2])]  # 7 top

    cube_faces = [[0, 1, 2, 3],  # bottom
                  [4, 5, 1, 0],  # front
                  [7, 6, 5, 4],  # top
                  [5, 6, 2, 1],  # right
                  [6, 7, 3, 2],  # back
                  [7, 4, 0, 3]]  # left

    obj = polyhedron(cube_points, cube_faces)
    return obj


def underdesk_drawer_vase() -> List[RenderTask]:
    drawer_z = 25
    layer_height = 0.2
    wall_thickness = 0.8
    back_wall_thickness = layer_height * 6
    front_a = 45 / 180 * pi
    front_joint_a = 23
    back_a = 70 / 180 * pi

    bottom_dim: P3 = (100., 75., wall_thickness)
    ridges_height = 2
    side_wall_ridges_height = ridges_height * 1.5
    ridges = 61
    center = False
    low_poly_mode = True
    bottom_front_offset = pi

    bottom = corrugated_cube_on_one_side(bottom_dim, offset=bottom_front_offset, ridges=ridges,
                                         ridges_height=ridges_height, center=center, low_poly_mode=low_poly_mode)
    bottom = intersection()(bottom, bevel_cube(mod_p3(bottom_dim, a=(None, None, ridges_height)),
                                               bevel_a=(0, 0, 0, front_a - front_joint_a / 180 * pi), center=center))

    front_top_offset = cos(front_a) * (drawer_z + bottom_dim[2] + ridges_height * 2)
    front_top_offset_extra_length = sqrt(pow(front_top_offset, 2) + pow(drawer_z, 2))
    front_dim = mod_p3(bottom_dim, y=front_top_offset_extra_length + wall_thickness * 4)
    front = corrugated_cube_on_one_side_front(front_dim, offset=bottom_front_offset, ridges=ridges,
                                              ridges_height=ridges_height, center=center, low_poly_mode=low_poly_mode)

    front = intersection()(front, bevel_cube(mod_p3(front_dim, az=ridges_height),
                                             bevel_a=(front_joint_a / 180 * pi, 0, 0, -front_a), center=center))
    front = front.rotate([front_a / pi * 180, 0, 0]).translate([0., bottom_dim[1], 0.])
    front_cutoff_dim = mod_p3(front_dim, az=ridges_height + 100, ax=(wall_thickness + ridges_height) * 2 + 10,
                              ay=wall_thickness + ridges_height + 20)
    front_cutoff = cube(front_cutoff_dim, center=center).translate(
        [-10, -wall_thickness - ridges_height, -ridges_height - 100])
    front_cutoff = front_cutoff.rotate([front_a / pi * 180, 0, 0]).translate([0., bottom_dim[1], 0.])

    back_top_offset = cos(back_a) * drawer_z
    back_top_extra_length = sqrt(pow(back_top_offset, 2) + pow(drawer_z, 2))
    slide_back_opening = 15
    back_wall_dim = mod_p3(bottom_dim, z=back_wall_thickness, y=back_top_extra_length, ax=-slide_back_opening)
    # back_wall = back_wall_round_sparse(back_wall_dim, center=center).translate([slide_back_opening / 2, 0, 0])
    back_wall = back_wall_round(back_wall_dim, center=center).translate([slide_back_opening / 2, 0, 0])
    back_wall0 = back_wall_round(back_wall_dim, center=center).translate([slide_back_opening / 2, 0, 0])
    back_wall += cube(mod_p3(back_wall_dim, ax=slide_back_opening, my=0.4), center=center) - back_wall0
    back_wall = back_wall.down(back_wall_dim[2]).rotate([180 - back_a / pi * 180, 0, 0]).translate([0, 0, 0])

    back_wall_cutoff_dim = [back_wall_dim[0] * 2, back_wall_dim[1] * 20, back_wall_dim[2] * 10]
    back_wall_cutoff = cube(back_wall_cutoff_dim, center=center).down(back_wall_dim[2]).translate(
        [-back_wall_cutoff_dim[0] / 2 + bottom_dim[0] / 2, -back_wall_cutoff_dim[1] / 2, back_wall_dim[2]]).rotate(
        [180 - back_a / pi * 180, 0, 0]).translate([0, 0, 0])

    side_wall_dim = (drawer_z, bottom_dim[1] + back_top_offset + wall_thickness + front_top_offset + 10, wall_thickness)
    side_walls = (
        corrugated_cube_on_one_side(side_wall_dim, offset=pi / 2, ridges=11, ridges_height=side_wall_ridges_height,
                                    center=False).translate([-side_wall_dim[0] - 0.5, -back_top_offset - wall_thickness,
                                                             -side_wall_dim[2] - side_wall_ridges_height]).rotate(
            [0, 90, 0]))

    side_walls += side_walls.mirrorX().translate([bottom_dim[0], 0, 0])

    top_cut_off = cube([500, 500, 100], center=False).translate([-250, -250, drawer_z])
    handle = cube(mod_p3(bottom_dim, y=drawer_z / 13, z=wall_thickness * 6), center=center).rotate(
        [-60, 0, 0]).translate([0, bottom_dim[1] + front_top_offset - 1, drawer_z - wall_thickness * 1])

    slide_original = (
        corrugated_cube_on_one_side(side_wall_dim, offset=pi / 2, ridges=11, ridges_height=side_wall_ridges_height,
                                    center=False))
    slide_thickness = 1.9
    tolerance = 0.4
    slide_inside = minkowski()(slide_original, sphere(r=tolerance / 2))

    slide_outside = intersection()(minkowski()(slide_original, sphere(r=slide_thickness / 2)), cube(
        mod_p3(side_wall_dim, az=side_wall_ridges_height + slide_thickness, ax=slide_thickness)).down(
        slide_thickness / 2).left(slide_thickness / 2))

    slide_cutout = cube(mod_p3(side_wall_dim, az=side_wall_ridges_height + slide_thickness, ax=slide_thickness)).up(
        side_wall_ridges_height).right(13)
    slide_cutout += cube(mod_p3(side_wall_dim, az=side_wall_ridges_height + slide_thickness, ax=slide_thickness)).right(
        22.5).down(slide_thickness)

    slide_a = 45 / 2
    slide = slide_outside - slide_inside - slide_cutout
    # slide = slide.rotate([0, 0, back_a / pi * 180])
    slide_cut = cube([100, 100, 100]).translate([-50, -100 + 1, -50]).rotate([0, 0, slide_a])
    slide_cut += cube([100, 100, 100]).translate([-50, -1, -50]).rotate([0, 0, -slide_a]).translate(
        [0, side_wall_dim[1], 0])
    # slide -= ~cube([100, 100, 100]).down(50).translate([-100 - 80, 0, 0])

    slides = (slide + slide.mirrorZ().up(10)) - slide_cut

    drawer = (bottom + front + back_wall + handle + (side_walls - front_cutoff)) - back_wall_cutoff - top_cut_off
    return [RenderTask(drawer, (0., 0., 0.), Path("underdesk_drawer")),
            RenderTask(slides, (-50., 0., 0.), Path("slide")),
            ]


def underdesk_drawer_vase_sets() -> List[RenderTask]:
    render_task: List[RenderTask] = list()
    render_task += underdesk_drawer_vase()

    return render_task


def main():
    cli_args = {'prog': 'MakerUnderdeskDrawerScript',
                'description': 'Creates the 3MFs and SCAD files for building the underdesk drawer vase',
                'default_output_path': Path(__file__).parent / '..' / 'build'}

    args, output_path, openscad_bin = solid2_utils_cli(*cli_args)

    render_tasks = underdesk_drawer_vase_sets()
    save_to_file(Path(output_path), openscad_bin, render_tasks, Path("all"),
                 include_filter_regex=re.compile(args.include_filter_regex or ""), verbose=args.verbose)


if __name__ == "__main__":
    main()
