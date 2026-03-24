"""
run.py
======

Command-line entry point for the ReLU region enumerator.

Usage
-----
    # Enumeration only
    python run.py --model NN_files/Arch3_2_96.pt --th 3.0 3.0 --output arch3

    # Enumeration + barrier certificate boundary cell extraction
    python run.py --model NN_files/Arch3_2_96.pt --th 3.0 3.0 --output arch3 --verification barrier --barrier-model NN_files/Arch3_2_96.pt
    python run.py --model NN_files/model_satellite_6d_16_16.pt --th 1.5 1.5 1.5 1.5 1.5 1.5 --output spacecraft --verification barrier --barrier-model NN_files/model_satellite_6d_16_16.pt
    python run.py --model NN_files/model_complex_3d_64_64.pt --th 3.0 3.0 3.0 --output Complex --verification barrier --barrier-model NN_files/model_complex_3d_64_64.pt
    python run.py --model NN_files/Decay_6d_12_12.pt --th 3.0 3.0 3.0 3.0 3.0 3.0 --output decay  --verification barrier --barrier-model NN_files/Decay_6d_12_12.pt

    # With profiling
    python run.py --model NN_files/Arch3_2_96.pt --th 3.0 3.0 --profile
    # Debugging
        python -m pdb run.py --model NN_files/Arch3_2_96.pt --th 3.0 3.0

Paper benchmarks
----------------
    python run.py --model NN_files/Arch3_2_96.pt --th 3.0 3.0 --output arch3
    python run.py --model NN_files/Decay_6d_12_12.pt --th 3.0 3.0 3.0 3.0 3.0 3.0 --output decay
    python run.py --model NN_files/model_satellite_6d_16_16.pt --th 1.5 1.5 1.5 1.5 1.5 1.5 --output spacecraft
    python run.py --model NN_files/model_quadrotor_lyapunov.pt --th 1.0 1.0 1.0 0.3 0.3 0.3 1.0 1.0 1.0 1.0 1.0 1.0 --output quadrotor
"""

import argparse
import cProfile
import os
import pstats
import sys

# Allow running directly after git clone without pip install.
sys.path.insert(0, os.path.dirname(__file__))

from relu_region_enumerator import enumeration_function


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enumerate polytopic linear regions of a ReLU neural network.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to TorchScript (.pt) controller network file.",
    )
    parser.add_argument(
        "--th",
        required=True,
        nargs="+",
        type=float,
        metavar="T",
        help="Per-dimension domain half-widths. Length must match network input dimension.",
    )
    parser.add_argument(
        "--output", "-o",
        default="result",
        help="Base name for output files (default: result).",
    )
    parser.add_argument(
        "--mode",
        default="Rapid_mode",
        choices=["Rapid_mode", "Low_Ram"],
        help="Enumeration mode (default: Rapid_mode).",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable the parallel JIT slicer.",
    )
    parser.add_argument(
        "--verification",
        default=None,
        choices=["barrier"],
        help="Verification task to run after enumeration. "
             "'barrier': extract cells straddling the b(x)=0 level set.",
    )
    parser.add_argument(
        "--barrier-model",
        default=None,
        metavar="PATH",
        help="Path to TorchScript (.pt) barrier certificate network. "
             "Required when --verification barrier is set.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Save cProfile statistics to <output>.prof.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    parallel = not args.no_parallel

    # Validate barrier args early.
    if args.verification == "barrier" and args.barrier_model is None:
        print("Error: --barrier-model is required when --verification barrier is set.")
        sys.exit(1)

    # Load barrier model if needed.
    barrier_model = None
    if args.barrier_model:
        import torch
        barrier_model = torch.jit.load(args.barrier_model, map_location="cpu")
        barrier_model.eval()
        print(f"Barrier model: {args.barrier_model}")

    print(f"Model        : {args.model}")
    print(f"Domain       : TH = {args.th}")
    print(f"Output       : {args.output}_polytope.h5")
    print(f"Mode         : {args.mode}  |  Parallel: {parallel}")
    print(f"Verification : {args.verification or 'none'}")
    print("-" * 50)

    kwargs = dict(
        NN_file=args.model,
        name_file=args.output,
        TH=args.th,
        mode=args.mode,
        parallel=parallel,
        verification=args.verification,
        barrier_model=barrier_model,
    )

    if args.profile:
        with cProfile.Profile() as pr:
            enumeration_function(**kwargs)
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        prof_file = args.output + ".prof"
        stats.dump_stats(filename=prof_file)
        print(f"\nProfiling data written to: {prof_file}")
        print(f"Visualise with:  snakeviz {prof_file}")
    else:
        enumeration_function(**kwargs)


if __name__ == "__main__":
    main()