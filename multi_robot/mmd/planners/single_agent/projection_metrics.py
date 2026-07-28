import json
import os
import time
import logging
import numpy as np
import torch

from mmd.common.eval_utils import collision_detect, calc_velocity

logger = logging.getLogger(__name__)


class ProjectionMetricsRecorder:
    """Per-projection-call metrics for the projector ablation, using the
    *exact same* violation definitions as the final reported metrics.

    Every ``projector.project()`` output is treated as if it were a final
    solution and pushed through the identical predicates that
    ``scripts/inference/inference_pcdiff.py`` uses to build
    ``processed_data.npz`` (and hence every number in the main tables):

        inter-robot : ``collision_detect(world_pos, safe_dist=2*robot_radius,
                      norm_order=2)``            -- WORLD frame, strict ``<``,
                                                    zero tolerance
        obstacle    : ``task.get_trajs_collision_and_free(world_traj,
                      num_interpolation=<same as run config>)`` -> per-waypoint
                      occupancy with ``margin=robot.radius`` -- WORLD frame,
                      zero tolerance
        velocity    : ``calc_velocity(norm_pos, dt=robot.dt) > vel_max +
                      VEL_TOL`` -- NORMALIZED frame, VEL_TOL = 5e-5

    The same three functions/objects the final pipeline calls are used here, so
    a violation counted by this recorder is a violation by the reported metric's
    own definition -- no tolerance band, no re-derived predicate.

    A *solver failure* is a (scene, denoising-step) solve whose projected
    trajectory has >= 1 inter-robot collision, >= 1 static-obstacle collision,
    or >= 1 velocity violation, under those definitions.

    Frames matter and are not interchangeable: the projectors work in
    normalized [-1, 1] space using a single scalar ``norm_scale =
    (2/pos_range).min()``, which is only exact when the position ranges of the
    two axes are equal. The collision predicates therefore run in WORLD
    coordinates, exactly as the final metrics do. The normalized -> world map
    is applied here directly rather than via
    ``dataset.unnormalize_trajectories``, because that method CLIPS to
    [-1, 1] and would hide out-of-bounds violations we are trying to measure.

    Batch layout matches the projectors: B = n_agents * G with batch index
    a * G + g; scene g couples the n_agents trajectories at group offset g.

    Results are appended as one JSON line per trial to ``out_path``.
    Enable by setting env var PROJ_METRICS_OUT=<path> (see ``from_env``).

    Set PROJ_METRICS_TRAJ_DIR to also dump the raw projector output of every
    call, so any future metric can be recomputed offline without re-running.
    One ``.npz`` per trial (written at finalize_trial, so a cancelled job keeps
    the trials it finished) holding normalized positions (n_steps, B, H, 2)
    plus the constants needed to map them to world coordinates.
    """

    def __init__(self, task, robot_radius, pos_mins, pos_range, n_agents,
                 out_path, vel_max, dt, vel_tol=5e-5, num_interpolation=0,
                 traj_dir=None):
        self.task = task
        self.robot_radius = float(robot_radius)      # WORLD units
        self.n_agents = int(n_agents)
        self.out_path = out_path
        self.vel_max = float(vel_max)
        self.dt = float(dt)
        self.vel_tol = float(vel_tol)
        self.num_interpolation = int(num_interpolation)
        self.traj_dir = traj_dir
        if self.traj_dir:
            os.makedirs(self.traj_dir, exist_ok=True)
        self._traj = []

        pm = torch.as_tensor(pos_mins).reshape(-1).float()
        pr = torch.as_tensor(pos_range).reshape(-1).float()
        self._pos_mins = pm
        self._pos_range = pr

        self._steps = []
        self._trial = 0
        logger.info(
            f'ProjectionMetricsRecorder (final-metric definitions): '
            f'r_world={self.robot_radius:.4f}, n_agents={self.n_agents}, '
            f'vel_max={self.vel_max} (+{self.vel_tol} tol), dt={self.dt:.6f}, '
            f'num_interpolation={self.num_interpolation}, out={self.out_path}, '
            f'traj_dir={self.traj_dir or "(disabled)"}')

    @classmethod
    def from_env(cls, task, robot_radius, pos_mins, pos_range, n_agents,
                 vel_max, dt, num_interpolation=0):
        """Build a recorder iff PROJ_METRICS_OUT is set; else return None."""
        out = os.environ.get('PROJ_METRICS_OUT', '').strip()
        if not out:
            return None
        vel_tol = float(os.environ.get('PROJ_METRICS_VEL_TOL', '5e-5'))
        traj_dir = os.environ.get('PROJ_METRICS_TRAJ_DIR', '').strip() or None
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        return cls(task=task, robot_radius=robot_radius, pos_mins=pos_mins,
                   pos_range=pos_range, n_agents=n_agents, out_path=out,
                   vel_max=vel_max, dt=dt, vel_tol=vel_tol,
                   num_interpolation=num_interpolation, traj_dir=traj_dir)

    def _to_world(self, x_norm):
        """LimitsNormalizer inverse for positions, WITHOUT the clip."""
        pr = self._pos_range.to(x_norm.device, x_norm.dtype)
        pm = self._pos_mins.to(x_norm.device, x_norm.dtype)
        return (x_norm + 1.0) / 2.0 * pr + pm

    def record(self, t_step, x_dofs, project_params, wall_s):
        """Record one projection call.

        Args:
            t_step: denoising step index (t_single).
            x_dofs: projector OUTPUT positions, tensor (B, H-1, 2), normalized.
            project_params: dict with 'pos_init' (B, 2) and 'dx_max'.
            wall_s: projector wall-clock seconds for this call (TimerCUDA).
        """
        t0 = time.time()
        n = self.n_agents
        B, Hm1, dim = x_dofs.shape
        G = B // n

        # Full trajectory incl. the pinned start -- the same object the final
        # metrics see (all_paths_array starts at the start state).
        pos_init = project_params['pos_init'][..., :dim].to(
            device=x_dofs.device, dtype=x_dofs.dtype)
        full_norm = torch.cat([pos_init[:, None, :], x_dofs.detach()], dim=1)
        H = full_norm.shape[1]

        # Raw projector output, kept verbatim for offline post-processing.
        if self.traj_dir is not None:
            self._traj.append(full_norm.cpu().numpy().astype(np.float32))

        # ---- velocity: inference_pcdiff.py:263 + analyze_velocity ----------
        # calc_velocity on NORMALIZED positions, ||dx||/dt vs vel_max + VEL_TOL
        pos_np = full_norm.cpu().numpy().astype(np.float64)
        vel = calc_velocity(pos_np, dt=self.dt)                  # (B, H-1)
        vel_thresh = self.vel_max + self.vel_tol
        vel_mask = vel > vel_thresh
        vel_scene_fail = vel_mask.reshape(n, G, H - 1).any(axis=(0, 2))
        vel = dict(
            viol=int(vel_mask.sum()), checked=int(B * (H - 1)),
            max_pen=float(max(float(vel.max()) - vel_thresh, 0.0)),
            scenes_fail=int(vel_scene_fail.sum()))

        # ---- world frame for the two collision predicates ------------------
        full_world = self._to_world(full_norm)
        world_np = full_world.cpu().numpy().astype(np.float64)

        # ---- inter-robot: inference_pcdiff.py:206 -------------------------
        # collision_detect(pos, safe_dist=2r, norm_order=2), strict '<'
        xs = world_np.reshape(n, G, H, dim).transpose(1, 0, 2, 3)  # (G,n,H,dim)
        rob_mask = collision_detect(
            xs, safe_dist=2.0 * self.robot_radius, norm_order=2)   # (G,n,n,H)
        rob_scene_fail = rob_mask.any(axis=(1, 2, 3))              # (G,)
        n_pairs = n * (n - 1) // 2
        # max penetration (diagnostic only; not part of the predicate)
        d = np.linalg.norm(xs[:, :, None] - xs[:, None, :], axis=-1)  # (G,n,n,H)
        iu = np.triu_indices(n, k=1)
        rob_max_pen = (float((2.0 * self.robot_radius - d[:, iu[0], iu[1]]).max())
                       if n_pairs else 0.0)
        rob = dict(
            viol=int(rob_mask.sum()) // 2,        # pairs counted twice
            checked=int(G * n_pairs * H),
            max_pen=max(rob_max_pen, 0.0),
            scenes_fail=int(rob_scene_fail.sum()))

        # ---- obstacle: inference_pcdiff.py:231 ----------------------------
        # task.get_trajs_collision_and_free -> compute_collision(margin=radius)
        trajs = full_world.to(**self.task.tensor_args)
        _, _, _, _, wp_coll = self.task.get_trajs_collision_and_free(
            trajs, return_indices=True,
            num_interpolation=self.num_interpolation)
        obs_mask = wp_coll.detach().cpu().numpy().astype(bool).reshape(n, G, -1)
        obs_scene_fail = obs_mask.any(axis=(0, 2))
        obs = dict(
            viol=int(obs_mask.sum()), checked=int(obs_mask.size),
            max_pen=0.0,   # occupancy is boolean; no signed depth available
            scenes_fail=int(obs_scene_fail.sum()))

        any_fail = vel_scene_fail | rob_scene_fail | obs_scene_fail
        self._steps.append(dict(
            t=int(t_step), wall_s=float(wall_s), n_scenes=int(G),
            obs=obs, rob=rob, vel=vel,
            scenes_fail_any=int(any_fail.sum()),
            dx_max=float(project_params['dx_max']),
            overhead_s=round(time.time() - t0, 4)))

    def finalize_trial(self):
        """Aggregate the accumulated steps into one JSONL line and reset."""
        if not self._steps:
            return
        steps = self._steps
        tot_solves = sum(s['n_scenes'] for s in steps)
        agg_types = {}
        for k in ('obs', 'rob', 'vel'):
            viol = sum(s[k]['viol'] for s in steps)
            checked = sum(s[k]['checked'] for s in steps)
            fails = sum(s[k]['scenes_fail'] for s in steps)
            agg_types[k] = dict(
                viol=viol, checked=checked,
                viol_rate=(viol / checked) if checked else 0.0,
                viol_per_solve=(viol / tot_solves) if tot_solves else 0.0,
                fail_rate=(fails / tot_solves) if tot_solves else 0.0,
                max_pen=max(s[k]['max_pen'] for s in steps))
        walls = [s['wall_s'] for s in steps]
        line = dict(
            trial=self._trial,
            n_steps=len(steps),
            n_scenes=steps[0]['n_scenes'],
            n_agents=self.n_agents,
            metric_defs='final',          # provenance: matches inference_pcdiff
            robot_radius_world=self.robot_radius,
            vel_max=self.vel_max,
            vel_tol=self.vel_tol,
            dt=self.dt,
            num_interpolation=self.num_interpolation,
            solver_fail_rate=(sum(s['scenes_fail_any'] for s in steps)
                              / tot_solves) if tot_solves else 0.0,
            proj_wall_s_total=float(np.sum(walls)),
            proj_wall_s_per_step=float(np.mean(walls)),
            agg=agg_types,
            steps=steps,
            ts=time.time(),
        )
        with open(self.out_path, 'a') as f:
            f.write(json.dumps(line) + '\n')

        # Dump this trial's raw projector outputs. Written per trial (not at
        # the end of the run) so a cancelled job keeps what it finished.
        if self.traj_dir is not None and self._traj:
            stem = os.path.splitext(os.path.basename(self.out_path))[0]
            np.savez_compressed(
                os.path.join(self.traj_dir,
                             f'{stem}_trial{self._trial:03d}.npz'),
                x_norm=np.stack(self._traj),          # (n_steps, B, H, 2)
                t_steps=np.array([s['t'] for s in steps], dtype=np.int32),
                wall_s=np.array(walls, dtype=np.float64),
                n_agents=self.n_agents,
                n_scenes=steps[0]['n_scenes'],
                pos_mins=self._pos_mins.numpy(),
                pos_range=self._pos_range.numpy(),
                robot_radius_world=self.robot_radius,
                vel_max=self.vel_max, vel_tol=self.vel_tol, dt=self.dt,
                num_interpolation=self.num_interpolation,
                trial=self._trial,
            )
        self._traj = []

        self._steps = []
        self._trial += 1
        logger.info(
            f'ProjectionMetrics trial {line["trial"]}: '
            f'fail_rate={line["solver_fail_rate"]:.3f}, '
            f'viol/solve obs={agg_types["obs"]["viol_per_solve"]:.2f} '
            f'rob={agg_types["rob"]["viol_per_solve"]:.2f} '
            f'vel={agg_types["vel"]["viol_per_solve"]:.2f}, '
            f'proj {line["proj_wall_s_per_step"]:.2f}s/step')
