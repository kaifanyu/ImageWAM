#!/usr/bin/env python
"""Does an estimated ``dz/dq`` actually predict the latent it is supposed to?

Section 18.9 of the ImageSTL Visual-SVPIO spec: before trusting a finite-
difference latent Jacobian to steer a robot, measure how well it predicts a
held-out joint perturbation.

For each estimation delta, ``J`` is built by central differences at the staged
start configuration, then for held-out random ``dq`` of several magnitudes it
reports

    relative error  ||dz_true - J dq|| / ||dz_true||
    cosine align    <dz_true, J dq> / (||dz_true|| ||J dq||)

The two numbers answer different questions and both matter.  Cosine alignment is
what the controller needs: ``dL/dq = J^T dL/dz`` is a descent direction as long
as the alignment is positive, however wrong the magnitude is.  Relative error is
what section 10.6 thresholds to decide when to re-probe; if it sits near 1 the
model has no usable scale and every step should be small and re-measured.

A single-joint block at the end reproduces the exact perturbation each column
was built from.  If *that* disagrees, the map is second-order at the probe
scale rather than the estimator being wrong.

    python -u -B experiments/libero/probe_latent_jacobian.py \
      --run-dir runs/multi_object_arm_preview \
      --editor-ae $FLUX2_AE_MODEL_PATH --flux2-src $FLUX2_SRC
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from latent_jacobian import JointLatentObserver, LatentJacobianEstimator  # noqa: E402
from sample_endpoint_trajectories import env_from_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-json")
    parser.add_argument(
        "--feature-encoder", choices=["flux_ae", "dinov3"], default="flux_ae"
    )
    parser.add_argument("--editor-ae")
    parser.add_argument("--flux2-src", default=str(REPO_ROOT / "third_party" / "flux2"))
    parser.add_argument("--dino-model", default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--latent-views", choices=["both", "agentview", "right", "wrist"], default="agentview"
    )
    parser.add_argument(
        "--fd-deltas", type=float, nargs="+", default=[0.005, 0.02, 0.05]
    )
    parser.add_argument(
        "--step-sizes", type=float, nargs="+", default=[0.002, 0.005, 0.01, 0.02, 0.05]
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.feature_encoder == "flux_ae" and not args.editor_ae:
        parser.error("--editor-ae is required with --feature-encoder flux_ae")

    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    start_state = np.load(run_dir / "start_state.npy")

    from score_endpoint_candidates import DinoV3FeatureMetric, FluxAutoencoderMetric

    if args.feature_encoder == "flux_ae":
        encoder: Any = FluxAutoencoderMetric(
            Path(args.editor_ae), Path(args.flux2_src), args.device
        )
    else:
        encoder = DinoV3FeatureMetric(args.dino_model, args.device)

    env = env_from_manifest(manifest)
    report: dict[str, Any] = {"held_out": [], "single_joint": []}
    try:
        env.seed(int(manifest["sim_seed"]))
        observer = JointLatentObserver(
            env,
            start_state,
            encoder,
            view_size=int(manifest["view_size"]),
            views=args.latent_views,
        )
        q0 = observer.home_q.copy()
        base = observer.observe(q0)
        z0 = base.descriptor.astype(np.float64)
        report["descriptor_dim"] = int(z0.size)
        report["token_shape"] = list(base.tokens.shape)
        report["num_joints"] = observer.num_joints
        print(f"descriptor dim {z0.size}  tokens {base.tokens.shape}")

        for fd_delta in args.fd_deltas:
            estimator = LatentJacobianEstimator(observer, delta_rad=float(fd_delta))
            jacobian, _ = estimator.central_difference(q0)
            print(
                f"\nfd_delta={fd_delta} rad  ||J||_F={np.linalg.norm(jacobian):.4e}  "
                f"cond={estimator.condition_number():.3e}"
            )
            generator = np.random.default_rng(args.seed)
            for scale in args.step_sizes:
                errors, alignments = [], []
                for _ in range(args.trials):
                    direction = generator.normal(size=observer.num_joints)
                    direction = direction / np.linalg.norm(direction) * float(scale)
                    q1 = observer.project_joint_limits(q0 + direction)
                    applied = q1 - q0
                    true_delta = observer.observe(q1).descriptor.astype(np.float64) - z0
                    predicted = jacobian @ applied
                    norm_true = np.linalg.norm(true_delta)
                    errors.append(
                        float(np.linalg.norm(true_delta - predicted) / (norm_true + 1e-12))
                    )
                    alignments.append(
                        float(
                            true_delta
                            @ predicted
                            / (norm_true * np.linalg.norm(predicted) + 1e-12)
                        )
                    )
                entry = {
                    "fd_delta_rad": float(fd_delta),
                    "step_norm_rad": float(scale),
                    "relative_error": float(np.mean(errors)),
                    "cosine_alignment": float(np.mean(alignments)),
                }
                report["held_out"].append(entry)
                print(
                    f"  |dq|={scale:.3f}  rel_err={entry['relative_error']:.3f}"
                    f"  cos_align={entry['cosine_alignment']:+.3f}"
                )

        estimator = LatentJacobianEstimator(observer, delta_rad=float(args.fd_deltas[0]))
        jacobian, _ = estimator.central_difference(q0)
        print("\nsingle joint, at the estimation delta:")
        for joint in range(observer.num_joints):
            step = np.zeros(observer.num_joints)
            step[joint] = float(args.fd_deltas[0])
            true_delta = observer.observe(q0 + step).descriptor.astype(np.float64) - z0
            predicted = jacobian @ step
            norm_true = np.linalg.norm(true_delta)
            entry = {
                "joint": joint,
                "relative_error": float(
                    np.linalg.norm(true_delta - predicted) / (norm_true + 1e-12)
                ),
                "cosine_alignment": float(
                    true_delta
                    @ predicted
                    / (norm_true * np.linalg.norm(predicted) + 1e-12)
                ),
            }
            report["single_joint"].append(entry)
            print(
                f"  joint {joint}: rel_err={entry['relative_error']:.3f}"
                f"  cos={entry['cosine_alignment']:+.3f}"
            )

        positive = [item["cosine_alignment"] > 0.0 for item in report["held_out"]]
        report["all_held_out_alignments_positive"] = bool(all(positive))
        print(
            "\nspec 28 Experiment B criterion "
            "(held-out perturbation predicted with positive latent-delta cosine): "
            f"{'PASS' if all(positive) else 'FAIL'}"
        )
    finally:
        env.close()

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n", "utf-8")


if __name__ == "__main__":
    main()
