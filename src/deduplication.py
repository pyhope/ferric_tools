#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yihang Peng
# Date: 05/05/2025

import os
import argparse
import numpy as np
import itertools
import shutil
import glob

import intro_ferric as fe
from c2d import read_poscar, fractional_to_cartesian

def is_similar(fp1, fp2, tol=1e-5, ratio_thresh=0.8):
    fp_copy = list(fp1)
    ufp_copy = list(fp2)

    i = 0
    while i < len(fp_copy):
        matched = False
        for j, b in enumerate(ufp_copy):
            if np.isclose(fp_copy[i], b, atol=tol):
                del fp_copy[i]
                del ufp_copy[j]
                matched = True
                break
        if not matched:
            i += 1

    i = 0
    while i < len(ufp_copy):
        matched = False
        for j, a in enumerate(fp_copy):
            if np.isclose(ufp_copy[i], a, atol=tol):
                del ufp_copy[i]
                del fp_copy[j]
                matched = True
                break
        if not matched:
            i += 1

    residual_ratio = (len(fp_copy) + len(ufp_copy)) / 2 / len(fp1)
    return residual_ratio <= (1 - ratio_thresh)

def filter_structures(input_dir, tol, threshold):
    vasp_files = sorted(glob.glob(os.path.join(input_dir, "*.vasp")))
    if not vasp_files:
        print(f"[Error] No .vasp files found in directory: {input_dir}")
        return

    fingerprints = []

    for filepath in vasp_files:
        data = read_poscar(filepath)
        coords_cart = (data["coordinates"] if data["coord_type"].lower() == "cartesian"
                       else fractional_to_cartesian(data["coordinates"], data["lattice"]))
        elements = data["elements"]
        counts = data["counts"]
        lattice = data["lattice"]

        element_list = []
        for elem, count in zip(elements, counts):
            element_list.extend([elem] * count)

        indices_fe = [j for j, e in enumerate(element_list) if e == 'Fe']
        if len(indices_fe) < 2:
            print(f"[Warn] Skipping {filepath}: less than 2 Fe atoms.")
            continue

        fe_distances = [fe.pbc_distance(coords_cart[a], coords_cart[b], lattice)
                        for a, b in itertools.combinations(indices_fe, 2)]
        fingerprint = tuple(sorted(fe_distances))
        fingerprints.append((filepath, fingerprint))
        print(f"[Info] {os.path.basename(filepath)}: fingerprint generated (len = {len(fingerprint)})")

    unique_fingerprints = []
    unique_filepaths = []

    for filepath, fp in fingerprints:
        if any(is_similar(fp, ufp, tol, threshold) for _, ufp in unique_fingerprints):
            continue
        unique_fingerprints.append((filepath, fp))
        unique_filepaths.append(filepath)

    print("=" * 80)
    print(f"[Result] Selected {len(unique_filepaths)} unique structures out of {len(fingerprints)}")

    output_dir = os.path.join(os.getcwd(), "selected")
    os.makedirs(output_dir, exist_ok=True)

    for new_idx, filepath in enumerate(unique_filepaths):
        dst_name = f"conf-{new_idx + 1}.vasp"
        dst_path = os.path.join(output_dir, dst_name)
        shutil.copy(filepath, dst_path)
        print(f"[Copy] {os.path.basename(filepath)} -> {dst_name}")

    print(f"[Done] Filtered structures saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter redundant .vasp structures by Fe-Fe fingerprint similarity.")
    parser.add_argument("input_dir", type=str, help="Directory containing .vasp files")
    parser.add_argument("--tol", type=float, default=1e-5, help="Distance tolerance for fingerprint comparison")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold (matched ratio)")

    args = parser.parse_args()

    filter_structures(args.input_dir, args.tol, args.threshold)
