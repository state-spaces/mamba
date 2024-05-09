import torch
import triton
from typing import Literal


def init(B, C, T, *, device, requires_grad=False):
    torch.manual_seed(12312323)
    gates = 0.999 + 0.001 * torch.rand(B, C, T, device=device, requires_grad=requires_grad)
    gates = gates.half().float()
    tokens = torch.rand(B, C, T, device=device, requires_grad=requires_grad)
    return gates, tokens


def make_benchmark(plot_name, *, direction, max_exponent=17):
    return triton.testing.Benchmark(
        x_names=["SEQUENCE_LENGTH"],  # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(7, max_exponent)],
        xlabel='sequence length',
        ylabel='ms',
        x_log=True,
        y_log=True,
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        #line_names=["triton", "ref", "warp"],
        #line_vals=["triton", "ref", "warp"],
        line_names=["warp"],
        line_vals=["warp"],
        plot_name=plot_name,
        args={
            "direction": direction,
        }
    )


def grad2(f, x, y, grad_out):
    grad = torch.autograd.grad(f(x, y), (x, y), grad_out)
    sum(x.sum().item() for x in grad)


def bench(provider, SEQUENCE_LENGTH, device="cuda", direction: Literal["forward", "backward", "train"] = "forward"):
    B, C, T = 8, 1536, SEQUENCE_LENGTH
    gates, tokens = init(B, C, T, device=device, requires_grad=direction=="train")
    outputs = torch.empty_like(tokens)
    grad_outputs = torch.empty_like(tokens)

    match provider:
        case "triton":
            print(f"Running {direction} {provider} with sequence length {SEQUENCE_LENGTH}")
            match direction:
                case "forward":
                    from accelerated_scan.triton import forward_scan
                    scan = lambda: forward_scan[(B,C)](gates, tokens, outputs, SEQUENCE_LENGTH, enable_fp_fusion=False)
                case "backward":
                    from accelerated_scan.triton import backward_scan
                    scan = lambda: backward_scan[(B,C)](gates, tokens, outputs, SEQUENCE_LENGTH, enable_fp_fusion=False)
                case "train":
                    # note that these measurements include time for memory allocation for forward output tensors
                    from accelerated_scan.triton import scan as train_scan
                    scan = lambda: grad2(train_scan, gates, tokens, grad_outputs)
        case "ref":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH} {direction}")
            from accelerated_scan.ref import scan as scan_ref
            match direction:
                case "forward":
                    scan = lambda: scan_ref(gates, tokens)
                case "backward":
                    scan = lambda: scan_ref(gates, tokens, reverse=True)
                case "train":
                    scan = lambda: grad2(scan_ref, gates, tokens, grad_outputs)
        case "warp":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH} {direction}")
            match direction:
                case "forward":
                    from accelerated_scan.warp import warpscan_forward
                    scan = lambda: warpscan_forward(gates, tokens, outputs, False)
                case "backward":
                    from accelerated_scan.warp import warpscan_forward
                    scan = lambda: warpscan_forward(gates, tokens, outputs, True)
                case "train":
                    # note that these measurements include time for memory allocation for forward output tensors
                    from accelerated_scan.warp import scan as train_scan
                    scan = lambda: grad2(train_scan, gates, tokens, grad_outputs)
        case _:
            raise ValueError(f"Unknown provider {provider}")

    # large warmup for benefit of torch.compile
    if direction == "train":
        ms = triton.testing.do_bench(scan, warmup=5000, rep=100)
    else:
        with torch.inference_mode():
            ms = triton.testing.do_bench(scan, warmup=5000, rep=100)
    return ms


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["forward", "backward", "train", "all"], default="all")
    args = parser.parse_args()

    directions = {
        'forward': make_benchmark("accelerated_scan: forward speed of (8,1536,seqlen), inference mode", direction="forward"),
        'backward': make_benchmark("accelerated_scan: backward speed of (8,1536,seqlen), inference mode", direction="backward"),
        'train': make_benchmark("accelerated_scan: training speed of (8,1536,seqlen)", direction="train", max_exponent=15),
    }

    benchmarks = []
    match args.direction:
        case "all":
            benchmarks.append(directions['forward'])
            benchmarks.append(directions['backward'])
            benchmarks.append(directions['train'])
        case dir:
            benchmarks.append(directions[dir])

    triton.testing.perf_report(benchmarks)(bench).run(save_path=".", print_data=True)
