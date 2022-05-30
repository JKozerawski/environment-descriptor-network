import sys
sys.path.insert(0, '../')

import argparse
import gc
import pickle
from typing import Final
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib
from tqdm import tqdm

from pathlib import Path
from lib.data import scenario_serialization
from lib.viz import scenario_visualization
from typing import Final, List, Optional, Sequence, Set, Tuple
from shapely.geometry import Point, Polygon, LineString, LinearRing
from shapely.affinity import affine_transform, rotate
import os
import math
import pandas as pd
matplotlib.use('agg')

from lib.data.data_schema import (
    ArgoverseScenario,
    ObjectType,
    Polyline,
    ScenarioStaticMap,
    TrackCategory,
)

_PlotBounds = Tuple[float, float, float, float]

# Configure constants
_OBS_DURATION_TIMESTEPS: Final[int] = 50
_PRED_DURATION_TIMESTEPS: Final[int] = 60

_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.0
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7
_PLOT_BOUNDS_BUFFER_M: Final[float] = 30.0

_DRIVABLE_AREA_COLOR: Final[str] = "#7A7A7A"
_LANE_SEGMENT_COLOR: Final[str] = "#E0E0E0"

_DEFAULT_ACTOR_COLOR: Final[str] = "#D3E8EF"
_FOCAL_AGENT_COLOR: Final[str] = "#ECA25B"
_AV_COLOR: Final[str] = "#007672"

_LANE_SEGMENT_COLOR: Final[str] = "#E0E0E0"
cities = ["austin", "miami", "pittsburgh", "dearborn", "washington-dc", "palo-alto"]


def normalize_trajectory(trajectory):
    start = trajectory[0]
    # First apply translation
    m = [1, 0, 0, 1, -start[0], -start[1]]
    ls = LineString(trajectory)
    # Now apply rotation, taking care of edge cases
    ls_offset = affine_transform(ls, m)
    end = ls_offset.coords[_OBS_DURATION_TIMESTEPS - 1]
    if end[0] == 0 and end[1] == 0:
        angle = 0.0
    elif end[0] == 0:
        angle = -90.0 if end[1] > 0 else 90.0
    elif end[1] == 0:
        angle = 0.0 if end[0] > 0 else 180.0
    else:
        angle = math.degrees(math.atan(end[1] / end[0]))
        if (end[0] > 0 and end[1] > 0) or (end[0] > 0 and end[1] < 0):
            angle = -angle
        else:
            angle = 180.0 - angle
    # Rotate the trajetory
    ls_rotate = rotate(ls_offset, angle, origin=(0, 0)).coords[:]
    # Normalized trajectory
    norm_xy = np.array(ls_rotate)
    return trajectory, norm_xy


def rotate_xyz(rad, centerX, centerY, points):
    cor_x2 = math.cos(rad) * (points[:, 0] - centerX) - math.sin(rad) * (points[:, 1] - centerY)
    cor_y2 = math.sin(rad) * (points[:, 0] - centerX) + math.cos(rad) * (points[:, 1] - centerY)
    points[:, 0] = cor_x2
    points[:, 1] = cor_y2
    return points

def plot_static_map_elements(static_map_elements: ScenarioStaticMap, show_ped_xings: bool = False, rad = 0, centerX=0, centerY=0, rotate=False) -> None:
    """Plot all static map elements associated with an Argoverse scenario.

    Args:
        static_map_elements: Static map elements to be plotted.
        show_ped_xings: Configures whether pedestrian crossings should be plotted.
    """
    # print("Plotting elements")
    # Plot drivable areas
    for drivable_area in static_map_elements["drivable_areas"].values():
        boundary_points = scenario_visualization._points_list_to_np(drivable_area["area_boundary"])
        if rotate:
            boundary_points = rotate_xyz(rad, centerX, centerY, boundary_points)
        plot_polygons([boundary_points], alpha=0.5, color=_DRIVABLE_AREA_COLOR)

    # Plot lane segments
    for lane_segment in static_map_elements["lane_segments"].values():
        left_lane_bounds = scenario_visualization._points_list_to_np(lane_segment["left_lane_boundary"])
        right_lane_bounds = scenario_visualization._points_list_to_np(lane_segment["right_lane_boundary"])
        if rotate:
            left_lane_bounds = rotate_xyz(rad, centerX, centerY, left_lane_bounds)
            right_lane_bounds = rotate_xyz(rad, centerX, centerY, right_lane_bounds)
        plot_polylines([left_lane_bounds, right_lane_bounds], linewidth=0.5, color=_LANE_SEGMENT_COLOR)

    # Plot pedestrian crossings
    if show_ped_xings:
        for ped_xing in static_map_elements["pedestrian_crossings"].values():
            edge1_points = scenario_visualization._points_list_to_np(ped_xing["edge1"])
            edge2_points = scenario_visualization._points_list_to_np(ped_xing["edge2"])
            if rotate:
                edge1_points = rotate_xyz(rad, centerX, centerY, edge1_points)
                edge2_points = rotate_xyz(rad, centerX, centerY, edge2_points)
            plot_polylines([edge1_points, edge2_points], alpha=1.0, color=_LANE_SEGMENT_COLOR)


def plot_polylines(polylines: Sequence[np.ndarray],
                    *,
                    style: str = "-",
                    linewidth: float = 1.0,
                    alpha: float = 1.0,
                    color: str = "r") -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    for polyline in polylines:
        plt.plot(polyline[:, 0], polyline[:, 1], style, linewidth=linewidth, color=color, alpha=alpha)

def plot_polygons(polygons: Sequence[np.ndarray], *, alpha: float = 1.0, color: str = "r") -> None:
    """Plot a group of filled polygons with the specified config.

    Args:
        polylines: Collection of polygons specified by (N, 2) arrays of vertices.
        alpha: Desired alpha for the polygon fill.
        color: Desired color for the polygon.
    """
    for polygon in polygons:
        plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)

def get_agent_trajectory(scenario):
    for track in scenario.tracks:
        if track.category == TrackCategory.FOCAL_TRACK:
            actor_trajectory = np.array(
                [list(object_state.position) for object_state in track.object_states])
    return actor_trajectory

def save_map(dataset, city, scenario_id, scenario, scenario_map_path, rotate):
    #rcParams['savefig.dpi'] = 300
    plt.figure(0, figsize=(8, 8))
    scenario_map = scenario_serialization.load_static_map_json(scenario_map_path)

    trajectory = get_agent_trajectory(scenario)
    trajectory, normalized_trajectory = normalize_trajectory(trajectory)

    obs_trajectory = trajectory[:_OBS_DURATION_TIMESTEPS]

    # Rotate:
    start_x = obs_trajectory[0, 0]
    start_y = obs_trajectory[0, 1]
    final_x = obs_trajectory[-1, 0]
    final_y = obs_trajectory[-1, 1]
    myradians = math.atan2(final_y - start_y, final_x - start_x)
    mydegrees = math.degrees(myradians)
    # rotate Agent path:
    cor_x = obs_trajectory[:, 0]
    cor_y = obs_trajectory[:, 1]
    rad = math.radians(90 - mydegrees)
    padding = 50
    x_min = final_x - padding
    x_max = final_x + padding
    y_min = final_y - padding
    y_max = final_y + padding
    centerX, centerY = x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2
    cor_x2 = math.cos(rad) * (cor_x - centerX) - math.sin(rad) * (cor_y - centerY)
    cor_y2 = math.sin(rad) * (cor_x - centerX) + math.cos(rad) * (cor_y - centerY)
    obs_trajectory = np.column_stack((cor_x2, cor_y2))

    plot_static_map_elements(scenario_map, True, rad, centerX, centerY, rotate=rotate)
    track_color = _FOCAL_AGENT_COLOR
    # Plot observable actor track:
    #plot_polylines([obs_trajectory], color=track_color, linewidth=2)
    # limit:
    x_min = obs_trajectory[-1, 0] - padding
    x_max = obs_trajectory[-1, 0] + padding
    y_min = obs_trajectory[-1, 1] - padding
    y_max = obs_trajectory[-1, 1] + padding
    plot_bounds = (x_min, x_max, y_min, y_max)
    plt.xlim(plot_bounds[0], plot_bounds[1])
    plt.ylim(plot_bounds[2], plot_bounds[3])
    #plt.gca().set_aspect("equal", adjustable="box")

    # Minimize plot margins and make axes invisible
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #plt.gca().axis('equal')
    plt.axis("off")
    flag = ""
    if not rotate:
        flag = "_base"
    if not os.path.exists("../intersections"+flag+"/" + dataset + "/" + city):
        os.makedirs("../intersections"+flag+"/" + dataset + "/" + city)
    plt.savefig("../intersections"+flag+"/" + dataset + "/" + city + "/" + str(scenario_id) + ".jpg", dpi=1*256/8)
    plt.cla()
    plt.clf()
    plt.close('all')
    plt.close()
    gc.collect()
    return trajectory, normalized_trajectory

def iterate_through_dataset(server=False, save_every=20000, map_only=False, rotate=False):

    main_path = "./"
    if server:
        main_path = "/data/jedrzej/argoverse2_raw/"
    sets = ["train", "val"]#, "test"]
    cities = ["austin", "miami", "pittsburgh", "dearborn", "washington-dc", "palo-alto"]


    for s in sets:
        counter = 0
        file_counter = 0

        all_scenario_files = sorted(Path(main_path+s).rglob("*.parquet"))
        for scenario_path in tqdm(all_scenario_files):
            if counter % save_every == 0 and not map_only:
                data = {}
                column_names = ["id", "input", "output", "normed_input", "normed_output"]
                for city in cities:
                    data[city] = pd.DataFrame(columns=column_names)
            try:
                scenario_id = scenario_path.stem.split("_")[-1]
                scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
                scenario_map = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
                trajectory, normalized_trajectory = save_map(s, scenario.city, scenario_id, scenario, scenario_map, rotate)
                if len(trajectory) == 110 and not map_only:
                    data[scenario.city] = data[scenario.city].append({"id": scenario_id, "input": trajectory[:_OBS_DURATION_TIMESTEPS], "output": trajectory[_OBS_DURATION_TIMESTEPS:],
                                          "normed_input": normalized_trajectory[:_OBS_DURATION_TIMESTEPS],
                                          "normed_output": normalized_trajectory[_OBS_DURATION_TIMESTEPS:]}, ignore_index=True)
                del scenario, scenario_map


            except:
                pass
            counter += 1
            if counter % save_every == 0 and not map_only:

                for city in cities:
                    if not os.path.exists("./data/"+s):
                        os.makedirs("./data/"+s)
                    pickle.dump(data[city], open("./data/"+s+"/"+city+"_"+str(file_counter)+ ".p", "wb"))
                file_counter += 1
        for city in cities:
            if not os.path.exists("./data/" + s):
                os.makedirs("./data/" + s)
            if not map_only:
                pickle.dump(data[city], open("./data/"+s+"/"+city+"_"+str(file_counter)+ ".p", "wb"))
        file_counter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", help="Run on server", action='store_true')
    parser.add_argument("--map_only", help="Save only map information", action='store_true')
    parser.add_argument("--rotate", help="Rotate map topology", action='store_true')
    parser.add_argument("--save_every", help="Frequency of saving a file", type=int, default=20000)

    args = parser.parse_args()
    iterate_through_dataset(server=args.server, save_every=args.save_every, map_only=args.map_only, rotate=args.rotate)

