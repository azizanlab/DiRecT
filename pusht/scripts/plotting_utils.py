import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from typing import List, Tuple
from tqdm import tqdm

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
        trajectories (np.ndarray): A NumPy array of shape (N, H, 2, I), where:
            - N is the total batch size, equal to (n_chunks * tup_size * B).
            - H is the full horizon (number of points per trajectory).
            - 2 represents (X, Y) coordinates.
            - I is the number of components, each plotted in a separate figure.
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
        raise ValueError("Input 'trajectories' must be of shape (P, N, H, 2, I).")

    P, N, H_full, _, I = trajectories.shape
    
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


def plot_collision_masks(collision_masks, safe_dist, safe_rates=None):
    """
    Plot collision masks as a grid where each row is a batch sample and each column is a horizon step.
    Green cells represent safe states, red cells represent collisions.
    
    Args:
        collision_masks: np.ndarray of shape (I, P, N, H) - boolean masks for collisions
        safe_dist: float - the safety distance threshold used for collision detection
    
    Returns:
        fig, axes - matplotlib figure and axes objects
    """
    I, P, N, H = collision_masks.shape
    # Aggregate across all players (OR): True if ANY player collides at that timestep
    # For P=2 this is equivalent to picking any single player (both had the same mask)

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
        
        # Convert the boolean mask to integers for plotting (OR across players)
        mask_to_plot = collision_masks[i].any(axis=0).astype(int)
        
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
            safe_rate = safe_rates[i].mean()  # average across players
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
        ax.axhline(hard_deck, color='red', linestyle=':', label=f'Constraint = {hard_deck:.4f}')
    ax.plot(t, v_max, label=rf'Max Velocity $\leq$ {v_max.max():.4f}', color='purple', linestyle='--')
    ax.plot(t, v_min, label='Min Velocity', color='green', linestyle='--')
    ax.axhline(v_mean_all, color='orange', linestyle='--', label=f'Overall Mean Velocity = {v_mean_all:.2f}')
    ax.set_title(subtitle, fontsize=14)
    ax.set_xlabel('Time Step (Horizon)')
    ax.set_ylabel('Velocity (units/s)')
    ax.legend(loc=legend_loc, fontsize='14')
    ax.grid()
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

