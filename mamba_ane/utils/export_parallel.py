# mamba_ane/utils/export_parallel.py
"""
Export StatefulMambaParallelHybrid to CoreML 9.0 (.mlpackage).

No ct.StateType — module is stateless.  Saves:
  outputs/StatefulMambaParallelHybrid_seq{L}_c{C}.mlpackage
  outputs/StatefulMambaParallelHybrid_seq{L}_c{C}.pt  (state_dict)

Usage:
  python mamba_ane/utils/export_parallel.py
  python mamba_ane/utils/export_parallel.py --seq-length 256 --num-classes 10
"""
import argparse, os, time
import numpy as np
import torch
import coremltools as ct

from mamba_ane.models.hybrid_parallel import StatefulMambaParallelHybrid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-classes",  type=int, default=1000)
    p.add_argument("--seq-length",   type=int, default=256)
    p.add_argument("--in-channels",  type=int, default=3)
    p.add_argument("--hidden-dim",   type=int, default=64)
    p.add_argument("--d-model",      type=int, default=256)
    p.add_argument("--d-state",      type=int, default=64)
    p.add_argument("--headdim",      type=int, default=64)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--out-dir",      default="./outputs")
    p.add_argument("--no-verify",    action="store_true")
    return p.parse_args()


def export(args):
    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"StatefulMambaParallelHybrid_seq{args.seq_length}_c{args.num_classes}"
    mlpackage_path = os.path.join(args.out_dir, f"{stem}.mlpackage")
    weights_path   = os.path.join(args.out_dir, f"{stem}.pt")

    print("=" * 70)
    print(f"StatefulMambaParallelHybrid → CoreML (seq={args.seq_length})")
    print("=" * 70)

    print(f"\n[1/5] Build model (seed={args.seed})")
    torch.manual_seed(args.seed)
    model = StatefulMambaParallelHybrid(
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        d_model=args.d_model,
        d_state=args.d_state,
        headdim=args.headdim,
    ).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {n_params/1e6:.2f}M")

    print(f"\n[2/5] Save weights → {weights_path}")
    torch.save(model.state_dict(), weights_path)

    print(f"\n[3/5] torch.jit.trace (input: (1, {args.in_channels}, {args.seq_length}))")
    shape = (1, args.in_channels, args.seq_length)
    with torch.no_grad():
        traced = torch.jit.trace(model, (torch.rand(*shape),))
    print("      Trace OK")

    print(f"\n[4/5] CoreML conversion (FLOAT16, iOS18, CPU_AND_NE)")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="x", shape=shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        # No states= — stateless model
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    print(f"      Conversion OK ({time.time()-t0:.1f}s)")

    print(f"\n[5/5] Save → {mlpackage_path}")
    mlmodel.save(mlpackage_path)

    if not args.no_verify:
        print("\n" + "-" * 70)
        print("Quick sanity check: 5 random inputs, PyTorch vs CoreML")
        print("-" * 70)
        torch.manual_seed(0)
        for i in range(5):
            x_np = np.random.rand(*shape).astype(np.float32)
            with torch.no_grad():
                pt_out = model(torch.from_numpy(x_np)).numpy()
            cml_out = mlmodel.predict({"x": x_np})["logits"]
            diff = np.abs(pt_out - cml_out).max()
            print(f"  Input {i}: max_abs = {diff:.2e}")

    print("\n" + "=" * 70)
    print("Export complete!")
    print("=" * 70)


if __name__ == "__main__":
    export(parse_args())
