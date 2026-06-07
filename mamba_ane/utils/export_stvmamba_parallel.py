# mamba_ane/utils/export_parallel.py
"""
Export STVMambaHybridANE to CoreML 9.0 (.mlpackage).

No ct.StateType — module is stateless.  Saves:
  outputs/STVMambaHybridANE_seq64_out64.mlpackage
  outputs/STVMambaHybridANE_seq64_out64.pt  (state_dict)

Usage:
  python mamba_ane/utils/export_parallel.py
"""
import argparse, os, time
import numpy as np
import torch
import coremltools as ct

from mamba_ane.models.stvmamba_hybrid import STVMambaHybridANE


def parse_args():
    p = argparse.ArgumentParser()
    # ICANSII Tri-plane specific arguments
    p.add_argument("--in-channels",  type=int, default=320) # e.g. 5 frames * 64 channels
    p.add_argument("--out-channels", type=int, default=64)  # Output tri-plane channels
    p.add_argument("--cond-dim",     type=int, default=16)  # IMU/Action conditioning vector
    p.add_argument("--patch-size",   type=int, default=16)  # 16x16 patch
    p.add_argument("--grid-size",    type=int, default=8)   # 8x8 grid (L = 64)
    
    # Mamba / Backbone arguments
    p.add_argument("--d-model",      type=int, default=256)
    p.add_argument("--d-state",      type=int, default=64)
    p.add_argument("--headdim",      type=int, default=64)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--out-dir",      default="./outputs")
    p.add_argument("--no-verify",    action="store_true")
    return p.parse_args()


def export(args):
    os.makedirs(args.out_dir, exist_ok=True)
    seq_length = args.grid_size * args.grid_size
    
    stem = f"STVMambaHybridANE_seq{seq_length}_out{args.out_channels}"
    mlpackage_path = os.path.join(args.out_dir, f"{stem}.mlpackage")
    weights_path   = os.path.join(args.out_dir, f"{stem}.pt")

    print("=" * 70)
    print(f"STVMambaHybridANE → CoreML (seq={seq_length})")
    print("=" * 70)

    print(f"\n[1/5] Build model (seed={args.seed})")
    torch.manual_seed(args.seed)
    model = STVMambaHybridANE(
        in_channels=args.in_channels,
        d_model=args.d_model,
        cond_dim=args.cond_dim,
        out_channels=args.out_channels,
        patch_size=args.patch_size,
        grid_size=args.grid_size,
        d_state=args.d_state,
        headdim=args.headdim,
    ).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {n_params/1e6:.2f}M")

    print(f"\n[2/5] Save weights → {weights_path}")
    torch.save(model.state_dict(), weights_path)

    # We now have TWO inputs: the buffer and the action condition
    shape_x = (1, args.in_channels, seq_length)
    shape_cond = (1, args.cond_dim)
    
    print(f"\n[3/5] torch.jit.trace (inputs: {shape_x} and {shape_cond})")
    with torch.no_grad():
        dummy_x = torch.rand(*shape_x)
        dummy_cond = torch.rand(*shape_cond)
        traced = torch.jit.trace(model, (dummy_x, dummy_cond))
    print("      Trace OK")

    print(f"\n[4/5] CoreML conversion (FLOAT16, iOS18, CPU_AND_NE)")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        # Explicitly map the two inputs for CoreML
        inputs=[
            ct.TensorType(name="triplane_buffer", shape=shape_x, dtype=np.float32),
            ct.TensorType(name="cond_emb", shape=shape_cond, dtype=np.float32)
        ],
        outputs=[ct.TensorType(name="pred_plane", dtype=np.float32)],
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
            x_np = np.random.rand(*shape_x).astype(np.float32)
            cond_np = np.random.rand(*shape_cond).astype(np.float32)
            
            with torch.no_grad():
                pt_out = model(torch.from_numpy(x_np), torch.from_numpy(cond_np)).numpy()
                
            cml_out = mlmodel.predict({
                "triplane_buffer": x_np, 
                "cond_emb": cond_np
            })["pred_plane"]
            
            diff = np.abs(pt_out - cml_out).max()
            print(f"  Input {i}: max_abs = {diff:.2e}")

    print("\n" + "=" * 70)
    print("Export complete!")
    print("=" * 70)


if __name__ == "__main__":
    export(parse_args())