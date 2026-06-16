import re
import os
import math
import yaml
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from ml_collections.config_dict import ConfigDict
from matplotlib.collections import LineCollection
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

# Helper function for safe extraction
def find_param(pattern, path, key, type_converter, default=None):
        match = re.search(pattern, path)
        if match:
            try:
                return type_converter(match.group(key))
            except (ValueError, IndexError):
                return default
        return default

def extract_coupling_cost_func(path: str) -> str:
    """
    Extracts the coupling method from a log-path string.

    Rules:
      - Base method: the part after the last hyphen in the 'max_vel-…-X' segment.
      - Coupling suffix: if the 'coupling' segment is written as 'coupling_ps', 'ps' is the suffix.
                        if it's just 'coupling', there's no suffix.
      - Final method is '<base>' if no coupling suffix, otherwise '<base>-<suffix>'.

    Examples:
    >>> s1 = "…/max_vel-max_v6.2-dpp/…/init_st50-max_vel_admm-coupling_ps-…"
    >>> extract_coupling_method(s1)
    'dpp-ps'

    >>> s2 = "…/max_vel-max_v10.789-exp_decay_l2_sqr/…/init_st10-max_vel_cvxpy-coupling-…"
    >>> extract_coupling_method(s2)
    'exp_decay_l2_sqr'
    """
    # 1) capture the method after the last hyphen in the max_vel segment
    m1 = re.search(r'max_vel-[^/]+-([^/]+)', path)
    if not m1:
        warnings.warn(f"No 'max_vel-…-X' pattern found in: {path!r}. Running backup search.")
        # raise ValueError(f"No 'max_vel-…-X' pattern found in: {path!r}")
        m1 = re.search(r"H\d+O\d+A\d+D\d+/([A-Za-z0-9_]+?)/max_stp*", path)
        if not m1:
            warnings.warn(f"No alternative pattern found in: {path!r}. Returning 'vanilla'.")
            return 'vanilla'
    base_method = m1.group(1)
    if not base_method:
        base_method = 'vanilla'

    # 2) capture optional coupling suffix after 'coupling'
    m2 = re.search(r'coupling(?:_([^-/]+))?(?:-|$)', path)
    if not m2:
        warnings.warn(f"No 'coupling' segment found in: {path!r}")
        # raise ValueError(f"No 'coupling' segment found in: {path!r}")
    suffix = m2.group(1)  # None if no '_…' part

    # 3) combine
    return base_method if suffix is None else f"{base_method}-{suffix}"


def extract_params_from_path(path: str) -> Dict[str, Any]:
    """
    Extracts key parameters from a directory path string.

    Args:
        path (str): The input directory path.

    Returns:
        Dict[str, Any]: A dictionary of extracted parameters.
    """
    params: Dict[str, Any] = {}

    # Diffusion configuration details
    m = re.search(r'pusht.*output/([^/]+?)/', path)
    if m:
        diff_cfg = m.group(1)
        params['diffusion_config'] = diff_cfg
        params['pred_horizon'] = find_param(r'H(\d+)', path, 1, int)
        params['action_horizon'] = find_param(r'A(\d+)',  path, 1, int, default=params.get('pred_horizon'))
        params['num_obs_steps'] = find_param(r'O(\d+)', path, 1, int, default=1)
        params['diffusion_timesteps'] = find_param(r'D(\d+)', path, 1, int, default=100)

    # Other parameters
    params['episode_length'] = find_param(r'max_stp(\d+)', path, 1, int)
    params['num_init_states'] = find_param(r'init_st(\d+)', path, 1, int)
    params['max_vel'] = find_param(r'max_v(\d+(?:\.\d+)?)', path, 1, float)
    params['projection_method'] = find_param(r'max_vel_([A-Za-z0-9_]+?)(?=-|/|$)', path, 1, str)
    params['num_trials'] = find_param(r'-t(\d+)(?:$|-)', path, 1, int)
    params['cost_function'] = extract_coupling_cost_func(path)

    # Stp and Scl parameters
    match = re.search(r'-stp(\d+)-scl([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', path)
    if match:
        params['stp'] = int(match.group(1))
        params['scl'] = float(match.group(2))

    return params


def extract_params_from_config(path: str, base_dir:str="") -> Dict[str, Any]:
    """
    Extracts key parameters from a configuration file path string.

    Args:
        path (str): The input configuration file path.

    Returns:
        Dict[str, Any]: A dictionary of extracted parameters.
    """
    assert os.path.isfile(path), f"Path does not point to a file: \'{path}\'"
    with open(path, 'r') as f:
        run_cfg_dict = yaml.safe_load(f)
        run_cfg = ConfigDict(run_cfg_dict)

    params: Dict[str, Any] = {}
    try:
        params['data_trained_models_dir'] = os.path.join(
            base_dir, 
            run_cfg.system.trained_models_dir_name, 
            run_cfg.environment.global_model_ids[0][0]
        )
        params['episode_length'] = run_cfg.planner.single_agent.horizon
        params['num_init_states'] = run_cfg.experiment.n_inits
        params['num_agents'] = run_cfg.experiment.num_agents
        params['num_trials'] = run_cfg.planner.single_agent.n_samples
        params['horizon'] = run_cfg.planner.single_agent.horizon
        params['model_id'] = run_cfg.environment.global_model_ids[0][0].split('-')[0]
        # params['model_id'] = run_cfg.environment.global_model_ids[0][0]
        params['dt'] = run_cfg.planner.single_agent.trajectory_duration / run_cfg.planner.single_agent.horizon
        params['robot_radius'] = run_cfg.robot.planar_disk_radius
        ## Projection
        # params['projection_method'] = find_param(r'max_vel_([A-Za-z0-9_]+?)(?=-|/|$)', path, 1, str)
        if run_cfg.planner.single_agent.planner_alg in ['pcd', 'projected']:
            params['projection_method'] = "admm"
        else:
            params['projection_method'] = 'none'
        params['max_vel'] = run_cfg.planner.projection.project_params.vel_max
        ## Coupling
        if run_cfg.planner.single_agent.planner_alg in ['pcd', 'coupling', 'coupling_ps']:
            params['cost_function'] = run_cfg.planner.single_agent.cost_func_key
            params['cost_func_param'] = run_cfg.planner.single_agent.cost_func_param
            params['stp'] = run_cfg.planner.single_agent.n_guide_steps
            params['scl'] = run_cfg.planner.single_agent.weight_grad_cost_constraints
        else:
            params['cost_function'] = 'vanilla'
        ## Guidance
        params['scl_obs_avoid_guide'] = run_cfg.planner.single_agent.weight_grad_cost_collision
    except RuntimeError as err:
        warnings.warn(f"Error extracting parameters from config: {err}")
        raise err

    return params


def plot_trajectories(
        trajectories: np.ndarray,
        title: str = "Trajectory Grids",
        cmap_name: str = 'viridis',
        xlim: tuple = None,
        ylim: tuple = None,
        figsize: tuple = None,
        wspace: float = 0.05,
        hspace: float = 0.05
    ):
    """
    Plots a batch of (X,Y) trajectories in a grid, where each grid cell
    corresponds to the last dimension of the input trajectories.
    Colors vary by time (horizon step) along each individual trajectory.
    A single colorbar is shown for the entire figure.

    Args:
        trajectories (np.ndarray): A NumPy array of shape (N, H, 2, I), where:
                                   - N is the batch size (number of trajectories per grid cell)
                                   - H is the horizon (number of points per trajectory)
                                   - 2 represents (X, Y) coordinates for each point.
                                   - I is the number of grid cells to plot.
        title (str): The super-title for the entire figure.
        cmap_name (str): The name of the matplotlib colormap.
        xlim (tuple, optional): Fixed (min, max) x-limits for each subplot. If None, auto-scaled per subplot.
        ylim (tuple, optional): Fixed (min, max) y-limits for each subplot. If None, auto-scaled per subplot.
        figsize (tuple, optional): Overall figure size (width, height). If None, automatically determined.
        wspace (float): Width space between subplots (fraction of subplot width).
        hspace (float): Height space between subplots (fraction of subplot height).
    """
    if trajectories.ndim != 4 or trajectories.shape[2] != 2:
        print("Error: Input 'trajectories' must be a NumPy array of shape (N, H, 2, I).")
        return None, None

    N_traj_per_grid, H, _, I_grids = trajectories.shape

    if I_grids == 0:
        print("Error: No grids to plot (I_grids=0).")
        return None, None
    if H <= 1 and N_traj_per_grid > 0:
        print("Warning: Horizon H <= 1, trajectories will not be visible as lines.")

    # Determine grid layout (square-ish)
    ncols = int(math.ceil(math.sqrt(I_grids)))
    nrows = int(math.ceil(I_grids / ncols))

    # Determine figure size
    _figsize = figsize
    if _figsize is None:
        subplot_width_default = 4
        subplot_height_default = 4
        max_total_width = 20
        max_total_height = 20

        fig_width = min(ncols * subplot_width_default, max_total_width)
        fig_height = min(nrows * subplot_height_default, max_total_height)
        _figsize = (fig_width, fig_height)

    fig, axes = plt.subplots(nrows, ncols, figsize=_figsize, squeeze=False)

    # Tighter spacing right after creating subplots
    fig.subplots_adjust(
        left=0.05, right=0.95,
        bottom=0.05, top=0.90,
        wspace=wspace, hspace=hspace
    )

    cmap = cm.get_cmap(cmap_name)
    if H > 1:
        norm_values = np.linspace(0, 1, H - 1)
        colors_for_one_trajectory = cmap(norm_values)
    else:
        colors_for_one_trajectory = np.array([])

    for i in range(I_grids):
        ax = axes[i // ncols, i % ncols]
        current = trajectories[:, :, :, i]

        if N_traj_per_grid == 0 or H <= 1:
            if N_traj_per_grid == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            if xlim: ax.set_xlim(xlim)
            if ylim: ax.set_ylim(ylim)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        starts = current[:, :-1, :]
        ends   = current[:, 1:, :]
        segments = np.stack((starts, ends), axis=2).reshape(-1, 2, 2)
        colors   = np.tile(colors_for_one_trajectory, (N_traj_per_grid, 1))

        lc = LineCollection(segments, colors=colors, linewidths=2)
        ax.add_collection(lc)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_aspect('equal', adjustable='box')

        # Limits
        if xlim is None or ylim is None:
            xs = current[:, :, 0]; ys = current[:, :, 1]
            mnx, mxx = xs.min(), xs.max()
            mny, mxy = ys.min(), ys.max()
            bx = (mxx - mnx) * 0.1 or 0.1
            by = (mxy - mny) * 0.1 or 0.1
            ax.set_xlim(xlim if xlim is not None else (mnx - bx, mxx + bx))
            ax.set_ylim(ylim if ylim is not None else (mny - by, mxy + by))
        else:
            ax.set_xlim(xlim); ax.set_ylim(ylim)

        ax.set_xticks([]); ax.set_yticks([])

    # Remove any extra axes
    for j in range(I_grids, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    fig.suptitle(title, fontsize=16)

    # Single colorbar
    if H > 1:
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=H - 1))
        sm.set_array([])
        cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Time (Horizon Step)")

    return fig, axes


def plot_trajectory_tuples(
    trajectories: np.ndarray,
    tup_size: int,
    h_start: int,
    h_end: int,
    title: str = "Trajectory Tuples",
    title_args: dict = {}, 
    cmap_name: str = 'viridis',
    xlim: tuple = None,
    ylim: tuple = None,
    figsize: tuple = None,
    wspace: float = 0.05,
    hspace: float = 0.05
) -> List[Tuple[plt.Figure, np.ndarray]]:
    """Plots trajectories organized into tuples, with chunks and batches merged.

    This function visualizes trajectories from a stacked NumPy array. The data
    in the first dimension (N) is assumed to be ordered by chunk, then by
    tuple member, then by batch instance. The chunk and batch dimensions are
    merged to create a single grid of subplots. A separate figure is generated
    for each component along the last dimension (I). Each subplot shows
    'tup_size' trajectories.

    Args:
        trajectories (np.ndarray): A NumPy array of shape (N, I, P, H, 2), where:
            - N is the total batch size, equal to (n_chunks * tup_size * B).
            - I is the number of components, each plotted in a separate figure.
            - P is the number of agetns.
            - H is the full horizon (number of points per trajectory).
            - 2 represents (X, Y) coordinates.
        tup_size (int): The number of trajectories that form a single tuple,
            plotted together in one subplot.
        n_chunks (int): The number of chunks the input 'trajectories' array
            is conceptually divided into along the N dimension.
        h_start (int): The starting time step to slice from the horizon H.
        h_end (int): The ending time step to slice from the horizon H.
        title (str): The base super-title for the figures. Each figure title
            will be appended with its component index.
        cmap_name (str): The name of the matplotlib colormap for the lines.
        xlim (tuple, optional): Fixed (min, max) x-limits for all subplots.
            If None, limits are auto-scaled for each subplot.
        ylim (tuple, optional): Fixed (min, max) y-limits for all subplots.
            If None, limits are auto-scaled for each subplot.
        figsize (tuple, optional): Overall figure size (width, height) for each
            figure. If None, it's automatically determined.
        wspace (float): The width space between subplots.
        hspace (float): The height space between subplots.

    Returns:
        List[Tuple[plt.Figure, np.ndarray]]: A list of tuples, where each
        tuple contains the matplotlib Figure and Axes objects for one
        component (i in I).
    """
    # --- Input Validation ---
    if trajectories.ndim != 5 or trajectories.shape[3] != 2:
        raise ValueError("Input 'trajectories' must be of shape (N, I, P, H, 2).")
    
    trajectories = trajectories.transpose(2, 0, 3, 4, 1)  # Rearrange to (P, N, H, 2, I)

    P, N, H_full, _, I = trajectories.shape  # (P, N, H, 2, I)
    
    if N == 0 or I == 0:
        print("Warning: No data to plot (N=0 or I=0).")
        return []

    if tup_size <= 0:
        raise ValueError("'tup_size' and 'n_chunks' must be positive integers.")

    total_subplots = N

    if not (0 <= h_start < h_end <= H_full):
        raise ValueError("Invalid slicing with h_start and h_end.")

    # --- Data Preparation ---
    # Slice the trajectories along the time horizon.
    traj_sliced = trajectories[:, :, h_start:h_end, :2, :]
    H = traj_sliced.shape[2]
    
    if H <= 1:
        print("Warning: Sliced horizon H <= 1, lines will not be visible.")

    grouped_traj = traj_sliced

    # --- Plotting ---
    output_figs = []
    cmap = cm.get_cmap(cmap_name)

    # Prepare the color array for a single trajectory. This is reused.
    if H > 1:
        norm_values = np.linspace(0, 1, H - 1)
        colors_for_one_trajectory = cmap(norm_values)
    else:
        colors_for_one_trajectory = np.array([])
    
    # Generate one figure for each component `i` in `I`.
    for i in tqdm(range(I)):
        # Data for the current figure, shape (total_subplots, tup_size, H, 2)
        data_for_fig_i = grouped_traj[:, :, :, :, i]

        # Determine grid layout (square-ish) for the subplots.
        if total_subplots > 0:
            ncols = int(math.ceil(math.sqrt(total_subplots)))
            nrows = int(math.ceil(total_subplots / ncols))
        else:
            nrows, ncols = 1, 1

        # Determine figure size automatically if not provided.
        _figsize = figsize
        if _figsize is None:
            subplot_w, subplot_h = 4, 4
            max_w, max_h = 24, 24
            fig_width = min(ncols * subplot_w, max_w)
            fig_height = min(nrows * subplot_h, max_h)
            _figsize = (fig_width, fig_height)
        
        fig, axes = plt.subplots(
            nrows, ncols, figsize=_figsize, squeeze=False
        )
        fig.subplots_adjust(
            left=0.05, right=0.9, bottom=0.05, top=0.9,
            wspace=wspace, hspace=hspace
        )

        # Iterate through all subplots.
        for i_subplot in range(total_subplots):
            r, c = i_subplot // ncols, i_subplot % ncols
            ax = axes[r, c]
            
            # Get the data for the current subplot, shape (tup_size, H, 2)
            current_tuple = data_for_fig_i[:, i_subplot, :, :]
            
            if H <= 1:
                ax.text(0.5, 0.5, "H<=1", ha='center', va='center')
            else:
                # Prepare segments for LineCollection.
                starts = current_tuple[:, :-1, :]
                ends = current_tuple[:, 1:, :]
                segments = np.stack((starts, ends), axis=2).reshape(-1, 2, 2)
                colors = np.tile(colors_for_one_trajectory, (tup_size, 1))
                lc = LineCollection(segments, colors=colors, linewidths=1.5)
                ax.add_collection(lc)

            # --- Aesthetics and Limits ---
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([])
            ax.set_yticks([])
            
            if xlim is None or ylim is None:
                if H > 0:
                    xs, ys = current_tuple[..., 0], current_tuple[..., 1]
                    mnx, mxx = xs.min(), xs.max()
                    mny, mxy = ys.min(), ys.max()
                    bx = (mxx - mnx) * 0.1 or 0.1
                    by = (mxy - mny) * 0.1 or 0.1
                    ax.set_xlim(xlim if xlim is not None else (mnx - bx, mxx + bx))
                    ax.set_ylim(ylim if ylim is not None else (mny - by, mxy + by))
            else:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        
        # Hide unused axes
        for j in range(total_subplots, nrows * ncols):
            fig.delaxes(axes.flatten()[j])

        title_str = f"{title}: Intial State {i}" 
        if len(title_args) > 0:
            title_str += '\n' + '(' + ", ".join(f"{k}={v}" for k, v in title_args.items()) + ')'
        fig.suptitle(title_str, fontsize=14)


        # Add a single, shared colorbar to the figure.
        if H > 1:
            sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=h_start, vmax=h_end-1))
            sm.set_array([])
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(f"Time Step ({h_start} to {h_end-1})")

        output_figs.append((fig, axes))

    return output_figs


def plot_collision_masks(collision_masks:np.ndarray, safe_dist, safe_rates=None):
    """
    Plot collision masks as a grid where each row is a batch sample and each column is a horizon step.
    Green cells represent safe states, red cells represent collisions.
    
    Args:
        collision_masks: np.ndarray of shape (N, I, P, P, H) - boolean masks for collisions
        safe_dist: float - the safety distance threshold used for collision detection
    
    Returns:
        fig, axes - matplotlib figure and axes objects
    """
    raise NotImplementedError("This function is not implemented yet.")
    N, I, P, _, H = collision_masks.shape
    collision_masks = collision_masks.transpose(1, 2, 0, 3)  # Rearrange to (I, P, N, H)
    # Just use player 0 since both players have the same collision patterns
    player_idx = 0

    # Create a figure with subplots. One for each initial state.
    fig_rows = int(np.ceil(I / 2))
    fig_cols = 2
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(12, fig_rows * 3), squeeze=False)
    fig.suptitle(f"Collision Masks by Initial State\n(Red=Collision, Green=Safe, Safe Dist={safe_dist})", fontsize=16)

    # Define a colormap: green for 0 (safe), red for 1 (collision)
    cmap = plt.cm.colors.ListedColormap(['green', 'red'])

    for i in range(I):
        row_idx = i // 2
        col_idx = i % 2
        ax = axes[row_idx, col_idx]
        
        # Convert the boolean mask to integers for plotting
        mask_to_plot = collision_masks[i, player_idx].astype(int)
        
        # Display the data as an image, where each pixel corresponds to a boolean value
        im = ax.imshow(mask_to_plot, cmap=cmap, aspect='auto', interpolation='none')
        
        ax.set_title(f"Initial State {i}")
        ax.set_xlabel("Horizon Step")
        ax.set_ylabel("Batch Sample")

        # Add text showing collision rate
        if safe_rates is None:
            collision_rate = np.mean(mask_to_plot)
            safe_rate = 1.0 - collision_rate
        else:
            safe_rate = safe_rates[i, 0]
        ax.text(0.98, 0.02, f"Safe: {safe_rate:.2%}", 
                transform=ax.transAxes, ha='right', va='bottom', 
                bbox=dict(facecolor='white', alpha=0.5))

    # Remove any extra axes if odd number of initial states
    if I % 2 != 0:
        fig.delaxes(axes[fig_rows-1, 1])

    # Add tight layout to ensure proper spacing and prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    return fig, axes


def plot_velocity_stats(
    t, v_mean, v_std, v_max, v_min, v_mean_all, hard_deck, 
    title='(Normalized) Velocity Statistics Over Horizon', 
    subtitle='',
    legend_loc='lower right',
):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.plot(t, v_mean, label='Mean Velocity', color='blue')
    ax.fill_between(
        t, 
        v_mean - v_std, 
        v_mean + v_std, 
        color='blue', alpha=0.2, label='Std Dev'
    )
    if hard_deck is not None and np.isfinite(hard_deck):
        ax.axhline(hard_deck, color='red', linestyle='-', label=f'Constraint = {hard_deck:.4f}')
    ax.plot(t, v_max, label=rf'Max Velocity $\leq$ {v_max.max():.4f}', color='purple', linestyle='--')
    ax.plot(t, v_min, label='Min Velocity', color='green', linestyle='--')
    ax.axhline(v_mean_all, color='orange', linestyle=':', label=f'Overall Mean Velocity = {v_mean_all:.4f}')
    ax.set_title(subtitle, fontsize=14)
    ax.set_xlabel('Time Step (Horizon)')
    ax.set_ylabel('Velocity (units/s)')
    ax.legend(loc=legend_loc, fontsize='14')
    ax.grid(alpha=0.5)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig, ax

def plot_velocity_histogram(
        v, bins=40, figsize=(10, 6), quantiles=[], colors=None, 
        title='Histogram of (Normalized) Velocities', 
        subtitle=''
    ):
    """
    Plot a histogram of velocities with optional quantile lines.

    Args:
        v (np.ndarray): Velocity array.
        bins (int): Number of histogram bins.
        figsize (tuple): Figure size.
        quantiles (list): List of quantiles to plot (e.g., [0.9, 0.95]).
        colors (list or None): List of colors for quantile lines.
    """
    fig, ax = plt.subplots(figsize=figsize)
    flat_v = v.flatten()
    ax.hist(flat_v, bins=bins, color='skyblue', edgecolor='black', alpha=0.8, density=True)
    if colors is None:
        # Use default matplotlib tab10 colors if not provided
        colors = matplotlib.colormaps['tab10'].colors
    for i, q in enumerate(quantiles):
        quantile_val = np.quantile(flat_v, q)
        color = colors[i % len(colors)]
        ax.axvline(quantile_val, color=color, linestyle='--', label=f'{int(q*100)}% Quantile: {quantile_val:.4f}')
    ax.set_title(subtitle, fontsize=14)
    ax.set_xlabel('Velocity (units/s)')
    ax.set_ylabel('Density')
    ax.grid(True, linestyle='--', alpha=0.5)
    if len(quantiles) > 0:
        ax.legend(fontsize=14)
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig, ax
