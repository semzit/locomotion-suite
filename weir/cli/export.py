from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from weir.cli.utils import add_checkpoint_arg, guarded_main
from weir.core.factory import create_algorithm

DEFAULT_OPSET = 17


def export_policy_to_onnx(
    policy: nn.Module,
    output_path: Path,
    *,
    obs_dim: int = 4,
    batch: int = 1,
) -> Path:
    """Export an inference policy to a standalone ONNX file with a dynamic batch axis."""
    output_path = Path(output_path)
    dummy_input = torch.randn(batch, obs_dim)
    policy.eval()
    with torch.no_grad():
        torch.onnx.export(
            policy,
            (dummy_input,),
            str(output_path),
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=DEFAULT_OPSET,
            dynamo=False,
        )
    return output_path


def verify_export(
    policy: nn.Module,
    onnx_path: Path,
    *,
    obs_dim: int = 4,
    batch: int = 1,
    samples: int = 5,
    seed: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> bool:
    """Return True when onnxruntime outputs match the PyTorch policy on seeded batches."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    policy.eval()
    with torch.no_grad():
        for _ in range(samples):
            obs_np = rng.standard_normal((batch, obs_dim), dtype=np.float32)
            torch_out = policy(torch.from_numpy(obs_np)).numpy()
            ort_out = cast(np.ndarray, session.run(None, {"obs": obs_np})[0])
            if not np.allclose(torch_out, ort_out, atol=atol, rtol=rtol):
                return False
    return True


def run_export(
    checkpoint: Path,
    output_path: Path,
    *,
    algo: str = "ppo",
    obs_dim: int = 4,
    batch: int = 1,
    samples: int = 5,
) -> Path:
    """Load a checkpoint, export its policy to ONNX, and verify onnxruntime parity."""
    algorithm = create_algorithm(algo)
    algorithm.load(Path(checkpoint))
    policy = algorithm.export_policy()
    if not isinstance(policy, nn.Module):
        raise TypeError(f"{type(policy).__name__} is not a torch.nn.Module")
    onnx_path = export_policy_to_onnx(policy, output_path, obs_dim=obs_dim, batch=batch)
    if not verify_export(policy, onnx_path, obs_dim=obs_dim, batch=batch, samples=samples):
        raise RuntimeError(
            f"ONNX model at {onnx_path} failed verification against the PyTorch policy"
        )
    return onnx_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a trained algorithm policy to a standalone ONNX file."
    )
    add_checkpoint_arg(parser)
    parser.add_argument(
        "--output", type=Path, default=Path("policy.onnx"), help="Output ONNX file path."
    )
    parser.add_argument("--algo", default="ppo", help="Algorithm name registered in the factory.")
    parser.add_argument(
        "--obs-dim", type=int, default=4, help="Observation dimension for the dummy input."
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size for the dummy input.")
    parser.add_argument(
        "--samples", type=int, default=5, help="Random batches used for onnxruntime verification."
    )

    def run(args: argparse.Namespace) -> dict[str, object]:
        output_path = run_export(
            args.checkpoint,
            args.output,
            algo=args.algo,
            obs_dim=args.obs_dim,
            batch=args.batch,
            samples=args.samples,
        )
        return {"checkpoint": str(args.checkpoint), "output": str(output_path)}

    return guarded_main(parser, run, "export")


if __name__ == "__main__":
    raise SystemExit(main())
