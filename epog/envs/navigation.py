import matplotlib.pyplot as plt
import numpy as np
from ai2thor.controller import Controller
from astar.search import AStar


class NavigationMap:
    def __init__(
        self, controller: Controller, robot_radius=0.25, grid_size=0.25
    ) -> None:
        self.controller = controller
        self.grid_size = grid_size
        self.robot_radius = robot_radius

        self.grid_map = self.init_grid_map()
        # self.show_grid_map()

    def convert_to_grid(self, x, z):
        x_grid = (x - self.x_origin) // self.grid_size
        z_grid = (z - self.z_origin) // self.grid_size
        return int(x_grid), int(z_grid)

    def init_grid_map(self):
        event = self.controller.step(action="GetReachablePositions")
        reachable_positions = event.metadata["actionReturn"]

        self.reachable_xs = [rp["x"] for rp in reachable_positions]
        self.reachable_zs = [rp["z"] for rp in reachable_positions]

        self.x_origin = min(self.reachable_xs) - self.grid_size
        self.z_origin = min(self.reachable_zs) - self.grid_size

        x_range, z_range = self.convert_to_grid(
            max(self.reachable_xs) + self.grid_size,
            max(self.reachable_zs) + self.grid_size,
        )

        grid_map = np.ones((x_range + 1, z_range + 1), dtype=np.int8)
        grid_map = grid_map.tolist()
        for rp in reachable_positions:
            x, z = self.convert_to_grid(rp["x"], rp["z"])
            grid_map[x][z] = 0
        return grid_map

    def find_nearest_node(self, x, y):
        x_l = []
        y_l = []
        for ix, iy in zip(self.reachable_xs, self.reachable_zs, strict=False):
            if x == ix and y == iy:
                continue
            x_l.append(ix)
            y_l.append(iy)
        dlist = [
            (x - ix) ** 2 + (y - iy) ** 2 for (ix, iy) in zip(x_l, y_l, strict=False)
        ]
        # except x, y
        minind = dlist.index(min(dlist))
        return x_l[minind], y_l[minind]

    def distance(self, start, goal, show=False) -> float:
        start_x, start_y = start
        goal_x, goal_y = goal
        sx, sy = self.find_nearest_node(start_x, start_y)
        gx, gy = self.find_nearest_node(goal_x, goal_y)
        grid_sx, grid_sy = self.convert_to_grid(sx, sy)
        grid_gx, grid_gy = self.convert_to_grid(gx, gy)
        full_path = AStar(self.grid_map).search((grid_sx, grid_sy), (grid_gx, grid_gy))
        moving_cost = 0
        for i in range(len(full_path) - 1):
            moving_cost += np.sqrt(
                (full_path[i + 1][0] - full_path[i][0]) ** 2
                + (full_path[i + 1][1] - full_path[i][1]) ** 2
            )
        if show:
            self.show_grid_map(grid_sx, grid_sy, grid_gx, grid_gy, full_path)
        return moving_cost * self.grid_size

    def show_grid_map(self, start_x, start_y, goal_x, goal_y, path=None):
        ox, oz = [], []
        for x in range(len(self.grid_map)):
            for z in range(len(self.grid_map[x])):
                if self.grid_map[x][z]:
                    ox.append(x)
                    oz.append(z)
        if path is not None:
            # draw arrow
            for i in range(len(path) - 1):
                plt.arrow(
                    path[i][0],
                    path[i][1],
                    path[i + 1][0] - path[i][0],
                    path[i + 1][1] - path[i][1],
                    head_width=1,
                    head_length=1,
                    fc="r",
                    ec="r",
                )
        plt.plot(ox, oz, ".k")
        plt.plot(start_x, start_y, "^r")
        plt.plot(goal_x, goal_y, "^c")
        plt.grid(True)
        plt.axis("equal")
        plt.show()
