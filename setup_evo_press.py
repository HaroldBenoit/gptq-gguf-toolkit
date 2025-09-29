#!/usr/bin/env python3
"""
setup_evo_press_lfm.py - Complete EvoPress setup for LFM2 model
Python translation of setup_evopress_lfm2_gptq.sh
"""

import json
from typing import Any


import os
import sys
import subprocess
import argparse
from pathlib import Path
import glob
from huggingface_hub import snapshot_download


def run_command(cmd, cwd=None, env=None):
    """Run a shell command and handle errors"""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            cwd=cwd,
            env=env,
            check=True,
            capture_output=False
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)


def download_model_if_not_local(model_path, local_dir=None):
    if not os.path.exists(model_path):
        snapshot_folder = snapshot_download(
            repo_id=model_path, local_dir_use_symlinks=False, local_dir=local_dir if local_dir else model_path
        )
        return snapshot_folder
    return model_path


def create_evopress_config_name(args):
    return f"evo-kl-gens-{args.generations}-configuration-{args.target_bitwidth}bit.txt"

def save_evopress_config(filename, args):
    evopress_args = {
        "fitness_fn": "kl",
        "generations": args.generations,
        "target_bitwidth": args.target_bitwidth,
        "offspring": args.offspring,
        "survivors_per_selection": args.survivors_per_selection,
        "tokens_per_selection": args.tokens_per_selection,
        "calibration_data": args.calibration_data,
        "calibration_tokens": args.calibration_tokens,
        "initially_generated": args.initially_generated,
        "initial_tokens": args.initial_tokens,
    }
    with open(filename, "w") as f:
        json.dump(evopress_args, f)

def main():
    parser = argparse.ArgumentParser(description="Complete EvoPress setup for any model")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--model-short-name", help="Short name for model (derived from model if not provided)")
    parser.add_argument("--target-bitwidth", type=float, default=4.2, help="Target bitwidth for quantization")
    parser.add_argument("--database-dir", help="Database directory (default: ./ep_database_<model_name>)")
    parser.add_argument("--quantized-dir", help="Quantized models directory (default: ./quantized_models_<model_name>)")
    parser.add_argument("--results-dir", help="Results directory (default: ./results_<model_name>)")
    parser.add_argument("--base-gguf-path", help="Path to base GGUF model (default: <model>.gguf)")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations for evolution")
    parser.add_argument("--offspring", type=int, default=128, help="Number of offspring per generation")
    parser.add_argument("--survivors-per-selection", nargs="+", type=int, default=[16, 4, 1], help="Survivors per selection")
    parser.add_argument("--tokens-per-selection", nargs="+", type=int, default=[2048, 16384, 131072], help="Tokens per selection")
    parser.add_argument("--calibration-data", default="LiquidAI/liquidtwo3", help="Calibration dataset")
    parser.add_argument("--calibration-tokens", type=int, default=2097152, help="Number of calibration tokens")
    parser.add_argument("--initially-generated", type=int, default=50, help="Number of initial candidates for non-integer bitwidth")
    parser.add_argument("--initial-tokens", type=int, default=4096, help="Number of tokens for initial selection")
    parser.add_argument("--not-imatrix", action="store_false", default=True, dest="imatrix", help=" imatrix for quantization")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    # Derive model short name from model path if not provided
    if args.model_short_name:
        model_short_name = args.model_short_name
    else:
        # Extract model name from path (e.g., "LiquidAI/LFM2-350M" -> "LFM2-350M")
        model_short_name = Path(args.model).name


    args.model = download_model_if_not_local(args.model, local_dir=os.path.join("checkpoints", model_short_name))

    # Set default directories based on model name
    database_dir = args.database_dir or f"./ep_database_{model_short_name.lower().replace('-', '_')}"
    quantized_dir = args.quantized_dir or f"./quantized_models_{model_short_name.lower().replace('-', '_')}" + ("_imatrix" if args.imatrix else "")
    results_dir = args.results_dir or f"./results_{model_short_name.lower().replace('-', '_')}"
    base_gguf_path = args.base_gguf_path

    if base_gguf_path is None or not os.path.exists(base_gguf_path):
        print(f"Base GGUF model not found, converting {args.model} to GGUF...")
        base_gguf_path = Path(f"{args.model}.gguf")

        if not os.path.exists(base_gguf_path):
            run_command([
                "python", "third_party/llama.cpp/convert_hf_to_gguf.py",
                "--outtype", "f16",
                "--outfile", str(base_gguf_path),
                args.model
            ])

    if args.imatrix:
        dataset_name = Path(args.calibration_data).name
        expected_calibration_data = f"{dataset_name}_calibration.txt"
        if not os.path.exists(expected_calibration_data):
            print(f"Calibration data not found, creating {expected_calibration_data}...")
            run_command([
                "python", "data/create_imatrix_calibration_data.py",
                "--model", args.model,
                "--calibration-data", args.calibration_data,
            ])

        expected_imatrix = f"{model_short_name}_{dataset_name}_imatrix.gguf"

        if not os.path.exists(expected_imatrix):
            print(f"Imatrix not found, creating {expected_imatrix}...")
            run_command([
                "third_party/llama.cpp/build/bin/llama-imatrix",
                "-m", str(base_gguf_path),
                "-f", str(expected_calibration_data),
                "-o", str(expected_imatrix),
            ])


    print(f"=== EvoPress Setup for {model_short_name} ===")
    print(f"Model: {args.model}")
    print(f"Target bitwidth: {args.target_bitwidth}")
    print(f"Database directory: {database_dir}")
    print(f"Quantized directory: {quantized_dir}")
    print(f"Results directory: {results_dir}")

    # Step 1: Create quantized models at different bitwidths
    print("\nStep 1: Creating quantized models at different bitwidths using GGUF quantization...")


    # Array of quantization levels to create
    quant_levels = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]

    full_path_quantized_dir: Any | Path = Path.cwd() / "quant" / "gguf" / quantized_dir

    to_quantize = True

    if full_path_quantized_dir.exists():
        if len(list(full_path_quantized_dir.glob("*.gguf"))) == len(quant_levels):
            to_quantize = False
            print("Quantized models already exist, skipping quantization...")

    if to_quantize:
        os.chdir("quant/gguf")
        Path(quantized_dir).mkdir(parents=True, exist_ok=True)


        # Use the GGUF quantization script

        run_command([
            "./run_quant.sh",
            str(base_gguf_path),
            " ".join(quant_levels),
            "--output-dir", quantized_dir,
            "--imatrix", str(expected_imatrix),
        ])


        os.chdir("../..")


    print("\nStep 2: Building EvoPress database...")

    full_path_database_dir: Any | Path = Path.cwd() / "mapper" / database_dir

    to_build_database = True

    if full_path_database_dir.exists():
        if len(list((full_path_database_dir / "models").glob("*.gguf"))) == len(quant_levels):
            to_build_database = False
            print("Database already exists, skipping build...")

    if to_build_database:

        # Step 2: Build EvoPress database
        os.chdir("mapper")


        # Collect all quantized models
        quantized_models = []
        for model_path in full_path_quantized_dir.glob("*.gguf"):
                quantized_models.append(str(model_path))

        # Build database if it doesn't exist
        if not os.path.exists(database_dir):
            print("Building database from quantized models...")
            cmd = ["./build_ep_database.sh", "--models"] + quantized_models + ["--output-dir", database_dir]
            run_command(cmd)
        else:
            print("Database already exists, skipping build...")

        os.chdir("..")



    configuration_name = create_evopress_config_name(args)
    save_evopress_config(full_path_database_dir / "layers-hf" / (configuration_name[:-len(".txt")] + "_config.json"), args)
    output_dir = full_path_database_dir / "layers-hf"
    full_configuration_path: Any | Path = output_dir / configuration_name

    # Step 3: Run EvoPress evolutionary search
    print("\nStep 3: Running EvoPress evolutionary search...")

    if not full_configuration_path.exists():
        os.chdir("evopress")

        # Create output directory for search results
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        print(f"Starting evolutionary search with target bitwidth: {args.target_bitwidth}...")

        cmd = [
            "python", "evo_quant_search.py",
            "--model_name_or_path", args.model,
            "--quant_weights_path", str(full_path_database_dir / "layers-hf"),
            "--target_bitwidth", str(args.target_bitwidth),
            "--generations", str(args.generations),
            "--offspring", str(args.offspring),
            "--initially_generated", str(args.initially_generated),
            "--initial_tokens", str(args.initial_tokens),
            "--survivors_per_selection"
        ]
        cmd.extend(map(str, args.survivors_per_selection))
        cmd.extend([
            "--tokens_per_selection"
        ])
        cmd.extend(map(str, args.tokens_per_selection))
        cmd.extend([
            "--calibration_data", args.calibration_data,
            "--calibration_tokens", str(args.calibration_tokens),
            "--fitness_fn", "kl",
            "--eval_datasets", "fineweb_edu", "wikitext2", "c4",
            "--output_dir", str(output_dir),
            "--configuration_name", str(configuration_name),
            "--eval_tokens", "1024"
        ])

        if not args.no_wandb:
            cmd.append("--log_wandb")

        run_command(cmd)

        os.chdir("..")

    hf_to_gguf_mapping_path = full_path_database_dir / "layers-hf" / "hf_to_gguf_mapping.json"
    with open(hf_to_gguf_mapping_path, "r") as f:
        hf_to_gguf_mapping = json.load(f)

    new_configuration=[]
    with open(full_configuration_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                key, value = line.split(":")
                key = key.strip()
                value = value.strip()
                new_key = hf_to_gguf_mapping[key+".weight"]
                new_configuration.append(f"{new_key}: {value}")

    new_full_configuration_path = full_path_database_dir / "layers-gguf" / configuration_name

    with open(new_full_configuration_path, "w") as f:
        for line in new_configuration:
            f.write(line + "\n")
    

    output_gguf_path = Path.cwd() / f"{model_short_name}-optimized-{configuration_name[:-len(".txt")]}.gguf"


    # Step 4: Convert configuration and prepare for assembly
    print("\nStep 4: Converting configuration for model assembly...")
    os.chdir("mapper")


    cmd = [
        "python", "gguf_stitcher.py",
        str(full_path_database_dir / "layers-gguf"),
        str(output_gguf_path),
        "--config", str(new_full_configuration_path),
        "--original-model", str(base_gguf_path.resolve()),
        "--default-quant-type", "F16",
    ]
    run_command(cmd)



    os.chdir("..")

    print("\nSetup completed successfully!")


if __name__ == "__main__":
    main()