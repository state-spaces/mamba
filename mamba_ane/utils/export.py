"""
mamba_ane/utils/export.py — Export StatefulMambaHybrid1D to CoreML 9.0 (.mlpackage)

Features:
  - Hybrid1D backbone (Conv1D + Linear)
  - MambaBlock state buffers (angle_state, ssm_state, k_state, v_state)
  - ct.StateType for persistent state across inferences
  - mlprogram format, iOS18+, FLOAT16 precision
  - Stateful API: make_state() + predict(state=...)
  - Optional numerical verification PyTorch vs CoreML

Usage (from repo root):
  python mamba_ane/utils/export.py
  python mamba_ane/utils/export.py --num-classes 10 --seq-length 224
  python mamba_ane/utils/export.py --ema-alpha 0.05 --no-verify
"""

import argparse
import os
import time

import numpy as np
import torch
import coremltools as ct

from mamba_ane.models.hybrid1d import StatefulMambaHybrid1D


def parse_args():
    p = argparse.ArgumentParser(description="Export StatefulMambaHybrid1D to CoreML")
    p.add_argument("--num-classes",         type=int,   default=1000)
    p.add_argument("--seq-length",          type=int,   default=224)
    p.add_argument("--backbone-hidden-dim", type=int,   default=256)
    p.add_argument("--backbone-output-dim", type=int,   default=512)
    p.add_argument("--mamba-d-state",       type=int,   default=64)
    p.add_argument("--mamba-headdim",       type=int,   default=64)
    p.add_argument("--mamba-num-heads",     type=int,   default=8)
    p.add_argument("--ema-alpha",           type=float, default=0.1)
    p.add_argument("--out-dir",             default="./exported_model")
    p.add_argument("--no-verify",           action="store_true")
    return p.parse_args()


def export(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("StatefulMambaHybrid1D -> CoreML 9.0 Export")
    print("=" * 70)

    print(f"\n[1/5] Build StatefulMambaHybrid1D")
    model = StatefulMambaHybrid1D(
        num_classes=args.num_classes,
        backbone_in_channels=3,
        backbone_hidden_dim=args.backbone_hidden_dim,
        backbone_output_dim=args.backbone_output_dim,
        seq_length=args.seq_length,
        ema_alpha=args.ema_alpha,
        mamba_d_state=args.mamba_d_state,
        mamba_headdim=args.mamba_headdim,
        mamba_num_heads=args.mamba_num_heads,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {n_params / 1e6:.2f}M")

    print(f"\n[2/5] Verify state buffers")
    buffers  = dict(model.named_buffers())
    required = ["mamba.angle_state", "mamba.k_state", "mamba.v_state", "mamba.ssm_state"]
    for buf in required:
        assert buf in buffers, f"Missing buffer: {buf}"
    print(f"      Buffers OK")

    print(f"\n[3/5] torch.jit.trace")
    shape = (1, 3, args.seq_length)
    with torch.no_grad():
        traced = torch.jit.trace(model, (torch.rand(shape),))
    print(f"      Trace OK")

    print(f"\n[4/5] CoreML 9.0 Conversion")
    states_list = [
        ct.StateType(wrapped_type=ct.TensorType(shape=model.mamba.angle_state.shape), name="mamba.angle_state"),
        ct.StateType(wrapped_type=ct.TensorType(shape=model.mamba.k_state.shape),     name="mamba.k_state"),
        ct.StateType(wrapped_type=ct.TensorType(shape=model.mamba.v_state.shape),     name="mamba.v_state"),
        ct.StateType(wrapped_type=ct.TensorType(shape=model.mamba.ssm_state.shape),   name="mamba.ssm_state"),
    ]
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="x", shape=shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        states=states_list,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    print(f"      Conversion OK ({time.time()-t0:.1f}s)")

    print(f"\n[5/5] Save model")
    model_name = (
        f"StatefulMambaHybrid1D_seq{args.seq_length}_c{args.num_classes}"
        f"_alpha{args.ema_alpha}"
    )
    out_path = os.path.join(args.out_dir, f"{model_name}.mlpackage")
    mlmodel.save(out_path)
    print(f"      Saved -> {out_path}")

    if not args.no_verify:
        print("\n" + "-" * 70)
        print("Numerical Verification: PyTorch vs CoreML")
        print("-" * 70)
        for buf in ["angle_state", "k_state", "v_state", "ssm_state"]:
            getattr(model.mamba, buf).zero_()
        coreml_state = mlmodel.make_state()
        for i in range(5):
            x_np = np.random.rand(1, 3, args.seq_length).astype(np.float32)
            with torch.no_grad():
                logits_pt = model(torch.from_numpy(x_np)).numpy()
            logits_cml = mlmodel.predict({"x": x_np}, state=coreml_state)["logits"]
            diff = np.abs(logits_pt - logits_cml).max()
            print(f"  Frame {i}: max diff = {diff:.2e}")

    print("\n" + "=" * 70)
    print("Export complete!")
    print("=" * 70)


if __name__ == "__main__":
    export(parse_args())
