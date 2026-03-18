"""
run.py
======

Command-line entry point for the ReLU region enumerator.

Usage
-----
    python run.py --model NN_files/Arch3_2_96.pt --th 3.0 3.0
    python run.py --model NN_files/Decay_6d_12_12.pt --th 3.0 3.0 3.0 3.0 3.0 3.0
    python run.py --model NN_files/model_satellite_6d_16_16.pt --th 1.5 1.5 1.5 1.5 1.5 1.5
    python run.py --model NN_files/model_quadrotor_modified.pt --th 1.0 1.0 1.0 0.3 0.3 0.3 1.0 1.0 1.0 1.0 1.0 1.0

Options
-------
    --model        Path to a TorchScript (.pt) model file.  [required]
    --th           Per-dimension domain half-widths.        [required]
    --output       Base name for output files. Default: result
    --mode         Rapid_mode (default) or Low_Ram.
    --no-parallel  Disable parallel JIT slicer.
    --profile      Save cProfile data to <output>.prof.
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
        help="Path to TorchScript (.pt) model file.",
    )
    parser.add_argument(
        "--th",
        required=True,
        nargs="+",
        type=float,
        metavar="T",
        help="Per-dimension domain half-widths. Must match the network input dimension.",
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
        "--profile",
        action="store_true",
        help="Save cProfile statistics to <output>.prof.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    parallel = not args.no_parallel

    print(f"Model  : {args.model}")
    print(f"Domain : TH = {args.th}")
    print(f"Output : {args.output}_polytope.h5")
    print(f"Mode   : {args.mode}  |  Parallel: {parallel}")
    print("-" * 50)

    if args.profile:
        with cProfile.Profile() as pr:
            enumeration_function(args.model, args.output, args.th, args.mode, parallel)
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        prof_file = args.output + ".prof"
        stats.dump_stats(filename=prof_file)
        print(f"\nProfiling data written to: {prof_file}")
        print(f"Visualise with:  snakeviz {prof_file}")
    else:
        enumeration_function(args.model, args.output, args.th, args.mode, parallel)


if __name__ == "__main__":
    main()