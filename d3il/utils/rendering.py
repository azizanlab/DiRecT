import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .arrays import to_np
from .video import save_video

from obstacles import PILLARS, NOVEL_CIRCULAR, NOVEL_PLANAR, CONSTRAINT_SETS


def save_single_trajectory_image(
    trajectory, filepath, constraint="", obstacle_margin=0.0, draw_obstacle_margin=False
):
    plotter = AvoidingTrajectoryPlotter(
        constraint=constraint,
        obstacle_margin=obstacle_margin,
        draw_obstacle_margin=draw_obstacle_margin,
    )
    fig, ax = plotter.setup_figure()
    plotter.plot_single_trajectory(trajectory, ax, style="multiple_real")
    plotter.add_environment_elements(ax)
    plotter.apply_legend(ax)
    plotter.save_figure(fig, filepath)


def save_trajectory_sequence(
    trajectories,
    base_path,
    folder_name,
    constraint="",
    obstacle_margin=0.0,
    draw_obstacle_margin=False,
):
    import os

    img_path = os.path.join(base_path, folder_name)
    os.makedirs(img_path, exist_ok=True)

    for step_i, trajectory in enumerate(trajectories):
        imgpath = os.path.join(img_path, f"{step_i}.png")
        save_single_trajectory_image(
            trajectory, imgpath, constraint, obstacle_margin, draw_obstacle_margin
        )


def save_prediction_overlay_plot(
    predicted_trajectories,
    real_trajectory,
    filepath,
    title=None,
    constraint="",
    obstacle_margin=0.0,
    draw_obstacle_margin=False,
):
    plotter = AvoidingTrajectoryPlotter(
        constraint=constraint,
        obstacle_margin=obstacle_margin,
        draw_obstacle_margin=draw_obstacle_margin,
    )
    fig, ax = plotter.plot_overlaid_predictions_and_real(
        predicted_trajectories,
        real_trajectory,
        title=title,
    )
    plotter.save_figure(fig, filepath)


def plot2img(fig, remove_margins=True):

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype="uint8").reshape((height, width, 4))


class AvoidingRenderer:

    def __init__(
        self,
        env_name,
        env,
        constraint="",
        obstacle_margin=0.0,
        draw_obstacle_margin=False,
    ):
        self.env_name = env_name
        self.env = env
        self.constraint = constraint
        self.obstacle_margin = obstacle_margin
        self.draw_obstacle_margin = draw_obstacle_margin

    def renders(self, observations):
        plotter = AvoidingTrajectoryPlotter(
            constraint=self.constraint,
            obstacle_margin=self.obstacle_margin,
            draw_obstacle_margin=self.draw_obstacle_margin,
        )
        fig, ax = plotter.setup_figure()
        plotter.plot_single_trajectory(observations, ax, style="multiple_real")
        plotter.add_environment_elements(ax)
        plotter.apply_legend(ax)
        img = plot2img(fig, remove_margins=False)
        plt.close(fig)
        return img

    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        diffusion_path = to_np(diffusion_path)
        n_diffusion_steps = diffusion_path.shape[0]
        frame = []
        for t in range(n_diffusion_steps):
            print(f"[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}")
            img = self.renders(diffusion_path[t])
            frame.append(img)
        frame = np.stack(frame, axis=0)
        save_video(savepath, frame, **video_kwargs)


class AvoidingTrajectoryPlotter:

    def __init__(self, constraint="", obstacle_margin=0.0, draw_obstacle_margin=False):
        self.constraint = constraint
        self.obstacle_margin = obstacle_margin
        self.draw_obstacle_margin = draw_obstacle_margin
        self.ax_limits = [[0.2, 0.8], [-0.3, 0.4]]
        self.figsize = (9, 9)
        self.fontsize_label = 16
        self.fontsize_ticks = 14
        self.fontsize_legend = 16
        self._legend_flags = {
            "target": False,
            "obstacle": False,
            "start": False,
            "end": False,
        }

    def setup_figure(self):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(self.ax_limits[0])
        ax.set_ylim(self.ax_limits[1])
        ax.set_facecolor([1, 1, 0.9])
        ax.set_aspect("equal", adjustable="box")

        fig.subplots_adjust(left=0.06, right=0.985, bottom=0.06, top=0.985)
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        ax.tick_params(axis="both", which="major", pad=1, labelsize=self.fontsize_ticks)
        ax.set_xlabel("X Position", fontsize=self.fontsize_label)
        ax.set_ylabel("Y Position", fontsize=self.fontsize_label)
        ax.grid(True, alpha=0.3)
        return fig, ax

    def add_environment_elements(self, ax):
        target_color = [0.4, 1, 0.4]

        x_min, x_max = self.ax_limits[0]
        y_min_target = 0.35
        y_max = self.ax_limits[1][1]
        if y_max > y_min_target:
            target_patch = patches.Rectangle(
                (x_min, y_min_target),
                x_max - x_min,
                y_max - y_min_target,
                facecolor=target_color,
                alpha=0.35,
                label="Target" if not self._legend_flags["target"] else "",
                zorder=0.5,
            )
            ax.add_patch(target_patch)
            self._legend_flags["target"] = True

        polytopic_constraints = self._get_polytopic_constraints(ax)
        self._add_polytopic_constraints(ax, polytopic_constraints)
        self._add_target_circles(ax)

    def _get_polytopic_constraints(self, ax):
        if self.constraint == "":
            return []
        elif self.constraint not in CONSTRAINT_SETS:
            raise ValueError(f"Unsupported constraint: {self.constraint}")

        for obs in NOVEL_CIRCULAR:
            center = list(obs.center)
            ax.add_patch(
                patches.Circle(
                    center,
                    obs.radius,
                    color="b",
                    alpha=0.3,
                    label="Obstacle" if not self._legend_flags["obstacle"] else "",
                )
            )
            self._legend_flags["obstacle"] = True

            if self.obstacle_margin > 0 and self.draw_obstacle_margin:
                self._add_margin_ring(ax, center, obs.radius, self.obstacle_margin, "b")

        return [
            [list(obs.point_a), list(obs.point_b), "below"]
            for obs in NOVEL_PLANAR
        ]

    def _add_polytopic_constraints(self, ax, polytopic_constraints):
        if self.constraint == "":
            return

        for i, constraint in enumerate(polytopic_constraints):
            mat = np.vstack((constraint[:2], np.zeros(2)))

            slope = (constraint[1][1] - constraint[0][1]) / (
                constraint[1][0] - constraint[0][0]
            )

            if slope > 0 and constraint[2] == "below":
                mat[2] = np.array([self.ax_limits[0][0], self.ax_limits[1][1]])
            elif slope < 0 and constraint[2] == "below":
                mat[2] = np.array([self.ax_limits[0][1], self.ax_limits[1][1]])
            elif slope > 0 and constraint[2] == "above":
                mat[2] = np.array([self.ax_limits[0][1], self.ax_limits[1][0]])
            elif slope < 0 and constraint[2] == "above":
                mat[2] = np.array([self.ax_limits[0][0], self.ax_limits[1][0]])

            clipped = self._clip_polygon_below_y(mat, 0.35)
            if len(clipped) >= 3:
                ax.add_patch(
                    patches.Polygon(
                        np.array(clipped),
                        color="b",
                        alpha=0.2,
                        label="Obstacle" if not self._legend_flags["obstacle"] else "",
                    )
                )
            self._legend_flags["obstacle"] = True

            if self.obstacle_margin > 0 and self.draw_obstacle_margin:
                self._add_margin_parallelogram(ax, constraint, "b")

    def _add_target_circles(self, ax):
        for idx, pillar in enumerate(PILLARS):
            center = list(pillar.center)
            radius = pillar.radius
            ax.add_patch(
                patches.Circle(
                    center,
                    radius,
                    color="r",
                    alpha=0.7,
                    label="Obstacle" if not self._legend_flags["obstacle"] else "",
                )
            )
            self._legend_flags["obstacle"] = True
            if self.obstacle_margin > 0 and self.draw_obstacle_margin:
                if not self._is_pillar_covered_by_obstacle(center):
                    self._add_margin_ring(ax, center, radius, self.obstacle_margin, "r")

    def _is_pillar_covered_by_obstacle(self, pillar_center):
        if self.constraint not in CONSTRAINT_SETS:
            return False
        for obs in NOVEL_CIRCULAR:
            distance = np.sqrt(
                (pillar_center[0] - obs.center[0]) ** 2
                + (pillar_center[1] - obs.center[1]) ** 2
            )
            if distance < obs.radius:
                return True
        return False

    def _add_margin_ring(self, ax, center, inner_radius, margin, color):
        outer_radius = inner_radius + margin

        outer_circle = patches.Circle(
            center, outer_radius, color=color, alpha=0.15, zorder=1
        )
        ax.add_patch(outer_circle)

        border_circle = patches.Circle(
            center,
            outer_radius,
            fill=False,
            edgecolor=color,
            alpha=0.7,
            linewidth=1.5,
            linestyle="--",
            zorder=3,
        )
        ax.add_patch(border_circle)

    def _add_margin_parallelogram(self, ax, constraint, color):
        p1, p2 = constraint[:2]
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        intercept = p2[1] - slope * p2[0]

        normal_x = -slope / np.sqrt(1 + slope**2)
        normal_y = 1 / np.sqrt(1 + slope**2)

        if constraint[2] == "below":
            margin_offset_x = -normal_x * self.obstacle_margin
            margin_offset_y = -normal_y * self.obstacle_margin
        else:
            margin_offset_x = normal_x * self.obstacle_margin
            margin_offset_y = normal_y * self.obstacle_margin

        original_mat = np.vstack((constraint[:2], np.zeros(2)))

        if slope > 0 and constraint[2] == "below":
            original_mat[2] = np.array([self.ax_limits[0][0], self.ax_limits[1][1]])
        elif slope < 0 and constraint[2] == "below":
            original_mat[2] = np.array([self.ax_limits[0][1], self.ax_limits[1][1]])
        elif slope > 0 and constraint[2] == "above":
            original_mat[2] = np.array([self.ax_limits[0][1], self.ax_limits[1][0]])
        elif slope < 0 and constraint[2] == "above":
            original_mat[2] = np.array([self.ax_limits[0][0], self.ax_limits[1][0]])

        margin_mat = original_mat.copy()
        margin_mat[0] += [margin_offset_x, margin_offset_y]
        margin_mat[1] += [margin_offset_x, margin_offset_y]

        margin_vertices = np.vstack([original_mat[:2], margin_mat[:2][::-1]])

        ax.add_patch(
            patches.Polygon(
                margin_vertices,
                color=color,
                alpha=0.1,
            )
        )

    def _clip_polygon_below_y(self, vertices, y_max):
        verts = [np.array(v, dtype=float).tolist() for v in vertices]
        if len(verts) == 0:
            return []
        output = []
        prev = verts[-1]
        prev_inside = prev[1] <= y_max
        for curr in verts:
            curr_inside = curr[1] <= y_max
            if curr_inside:
                if not prev_inside:
                    denom = curr[1] - prev[1]
                    if denom != 0:
                        t = (y_max - prev[1]) / denom
                        x_int = prev[0] + t * (curr[0] - prev[0])
                        output.append([x_int, y_max])
                output.append(curr)
            else:
                if prev_inside:
                    denom = curr[1] - prev[1]
                    if denom != 0:
                        t = (y_max - prev[1]) / denom
                        x_int = prev[0] + t * (curr[0] - prev[0])
                        output.append([x_int, y_max])
            prev, prev_inside = curr, curr_inside
        return output

    def plot_single_trajectory(
        self, trajectory, ax, style="actual", label_prefix="", show_labels=True
    ):
        x_coords = trajectory[:, 2]
        y_coords = trajectory[:, 3]

        if style == "actual":
            ax.plot(
                x_coords,
                y_coords,
                "k-",
                linewidth=3,
                label=f"{label_prefix}Actual Trajectory" if show_labels else "",
            )
            ax.plot(
                x_coords[0],
                y_coords[0],
                "go",
                markersize=10,
                markeredgecolor="darkgreen",
                markeredgewidth=1,
                label="Start" if not self._legend_flags["start"] else "",
            )
            self._legend_flags["start"] = True
            ax.plot(
                x_coords[-1],
                y_coords[-1],
                "bo",
                markersize=10,
                markeredgecolor="navy",
                markeredgewidth=1,
                label="End" if not self._legend_flags["end"] else "",
            )
            self._legend_flags["end"] = True
            if len(x_coords) > 2:
                ax.plot(
                    x_coords[1:-1],
                    y_coords[1:-1],
                    "ko",
                    markersize=3,
                    alpha=0.7,
                )

        elif style == "predicted":
            ax.plot(
                x_coords,
                y_coords,
                "m-",
                linewidth=1.5,
                alpha=0.4,
                label=f"{label_prefix}Predicted Trajectories" if show_labels else "",
            )
            ax.plot(
                x_coords[0],
                y_coords[0],
                "ms",
                markersize=5,
                alpha=0.5,
            )
            ax.plot(
                x_coords[-1],
                y_coords[-1],
                "m^",
                markersize=5,
                alpha=0.5,
            )
            if len(x_coords) > 2:
                ax.plot(
                    x_coords[1:-1],
                    y_coords[1:-1],
                    "mx",
                    markersize=3,
                    alpha=0.4,
                )

        elif style == "multiple_real":
            ax.plot(
                x_coords,
                y_coords,
                "k-",
                linewidth=2,
                label=f"{label_prefix}Trajectory" if show_labels else "",
            )
            ax.plot(
                x_coords[0],
                y_coords[0],
                "go",
                markersize=8,
                label="Start" if not self._legend_flags["start"] else "",
            )
            self._legend_flags["start"] = True
            ax.plot(
                x_coords[-1],
                y_coords[-1],
                "bo",
                markersize=8,
                label="End" if not self._legend_flags["end"] else "",
            )
            self._legend_flags["end"] = True
            if len(x_coords) > 2:
                ax.plot(
                    x_coords[1:-1],
                    y_coords[1:-1],
                    "co",
                    markersize=4,
                    alpha=0.6,
                )

    def plot_overlaid_predictions_and_real(
        self,
        predicted_trajectories,
        real_trajectory,
        title=None,
    ):
        fig, ax = self.setup_figure()

        for i, trajectory in enumerate(predicted_trajectories):
            self.plot_single_trajectory(
                trajectory, ax, style="predicted", show_labels=(i == 0)
            )

        self.plot_single_trajectory(real_trajectory, ax, style="actual")

        if title:
            ax.set_title(title, fontsize=self.fontsize_label)

        self.add_environment_elements(ax)

        self.apply_legend(ax)

        return fig, ax

    def _trajectory_violates(self, trajectory):
        """Check if any point in the trajectory violates constraints."""
        if self.constraint == "":
            return False
        from obstacles import check_violation
        for t in range(len(trajectory)):
            if check_violation(trajectory[t], self.constraint) > 0:
                return True
        return False

    def plot_multiple_real_trajectories(self, trajectories, title=None):
        fig, ax = self.setup_figure()

        for i, trajectory in enumerate(trajectories):
            x_coords = trajectory[:, 2]
            y_coords = trajectory[:, 3]

            violated = self._trajectory_violates(trajectory)
            color = "gray" if violated else "k"

            ax.plot(
                x_coords,
                y_coords,
                color=color,
                linestyle="-",
                linewidth=2,
                alpha=0.7,
                label="Trajectory" if i == 0 else "",
            )
            ax.plot(
                x_coords[0],
                y_coords[0],
                "go",
                markersize=8,
                label="Start" if not self._legend_flags["start"] else "",
            )
            self._legend_flags["start"] = True
            ax.plot(
                x_coords[-1],
                y_coords[-1],
                "bo",
                markersize=8,
                label="End" if not self._legend_flags["end"] else "",
            )
            self._legend_flags["end"] = True
            if len(x_coords) > 2:
                ax.plot(x_coords[1:-1], y_coords[1:-1], "co", markersize=4, alpha=0.6)

        self.add_environment_elements(ax)

        self.apply_legend(ax)

        return fig, ax

    def save_figure(self, fig, filepath, dpi=150, bbox_inches=None):
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
        print(f"Saved trajectory plot to: {filepath}")

    def apply_legend(self, ax):

        from matplotlib.legend_handler import HandlerTuple

        handles, labels = ax.get_legend_handles_labels()

        new_handles = []
        new_labels = []
        seen = set()
        for h, l in zip(handles, labels):
            if not l:
                continue
            if l.lower() == "obstacle":
                continue
            if l not in seen:
                seen.add(l)
                new_handles.append(h)
                new_labels.append(l)

        purple_proxy = patches.Rectangle((0, 0), 1, 1, facecolor="b", alpha=0.2)
        red_proxy = patches.Circle((0.5, 0.5), 0.5, facecolor="r", alpha=0.7)
        composite = (purple_proxy, red_proxy)

        new_handles.append(composite)
        new_labels.append("Obstacle")

        ax.legend(
            new_handles,
            new_labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc="lower left",
            fontsize=self.fontsize_legend,
        )
