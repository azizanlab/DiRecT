import re
import warnings
from typing import Dict, Any

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
      • Base method: the part after the last hyphen in the 'max_vel-…-X' segment.
      • Coupling suffix: if the 'coupling' segment is written as 'coupling_ps', 'ps' is the suffix.
                        if it's just 'coupling', there's no suffix.
      • Final method is '<base>' if no coupling suffix, otherwise '<base>-<suffix>'.

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

