#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yihang Peng
# Date: 05/05/2025

import numpy as np
import argparse
import csv
import random
from c2d import read_poscar, fractional_to_cartesian, cartesian_to_fractional

def pbc_distance(vec1, vec2, lattice):
    # Compute the shortest distance between two atoms under periodic boundary conditions
    delta = vec2 - vec1
    frac = np.dot(delta, np.linalg.inv(lattice.T))  # Convert to fractional coordinates
    frac -= np.round(frac)                          # Apply minimum image convention
    cart = np.dot(frac, lattice.T)                  # Convert back to Cartesian
    return np.linalg.norm(cart)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Charge-coupled cation substitution code.")
    parser.add_argument("--input_file", "-i", type=str, default="POSCAR", help="Input POSCAR file")
    parser.add_argument("--output_file", "-o", type=str, default="modified.vasp", help="Output modified structure file")
    parser.add_argument("--element_a", "-a", type=str, default="Si", help="Target atom A (default: Si)")
    parser.add_argument("--element_b", "-b", type=str, default="Fe", help="Neighbor atom B (default: Fe)")
    parser.add_argument("--element_c", "-c", type=str, default="Al", help="Replacement element C (default: Al)")
    parser.add_argument("--element_d", "-d", type=str, default="Mg", help="Atoms to sort after substitution (default: Mg)")
    parser.add_argument("--top_n", "-n", type=int, default=None, help="Number of A atoms to replace (default: half of B atoms)")
    parser.add_argument("--save_csv", "-s", action="store_true", default=False, help="Save replacement pairs to CSV file")
    parser.add_argument("--csv_filename", "-csv", type=str, default="replaced_pairs.csv", help="CSV file name (default: replaced_pairs.csv)")
    parser.add_argument("--disable_high_d2", "-dhd", action="store_true", help="Disable filtering by high d2 (second-nearest B distance)")
    parser.add_argument("--random_seed", "-rs", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load structure and convert coordinates if necessary
    data = read_poscar(args.input_file)
    lattice = data["lattice"]
    coord_type = data["coord_type"].lower()

    if coord_type == "cartesian":
        coords_cart = data["coordinates"]
    elif coord_type == "direct":
        print("Converting Direct coordinates to Cartesian...")
        coords_cart = fractional_to_cartesian(data["coordinates"], lattice)
    else:
        raise ValueError("Unknown coordinate type: must be Cartesian or Direct.")

    elements = data["elements"]
    counts = data["counts"]
    comments = data["comments"]

    # Build full element list
    element_list = []
    for elem, count in zip(elements, counts):
        element_list.extend([elem] * count)

    indices = list(range(len(element_list)))
    indices_a = [i for i, e in enumerate(element_list) if e == args.element_a]
    indices_b = [i for i, e in enumerate(element_list) if e == args.element_b]
    indices_d = [i for i, e in enumerate(element_list) if e == args.element_d]

    # Default replacement count: half the number of B atoms
    n_replace = args.top_n if args.top_n is not None else len(indices_b) // 2

    tol = 1e-3  # tolerance for distance equality
    all_pairs_info = []     # stores valid (d1, d2, A_index, B1_index) tuples
    d1_values = []          # list of d1 values
    a_d_distances = []      # distance from A to all D atoms (e.g., Mg)

    # Loop over each A atom and evaluate distances to B and D atoms
    for ia in indices_a:
        a_pos = coords_cart[ia]
        distances = []

        # Measure A–D (e.g., A–Mg) distances
        for id in indices_d:
            dist = pbc_distance(a_pos, coords_cart[id], lattice)
            a_d_distances.append(dist)

        # Measure A–B distances
        for ib in indices_b:
            b_pos = coords_cart[ib]
            dist = pbc_distance(a_pos, b_pos, lattice)
            distances.append((dist, ib))

        if not distances:
            continue

        # Sort and find the nearest neighbor B
        distances.sort()
        d1 = distances[0][0]
        nearest = [d for d in distances if abs(d[0] - d1) < tol]

        if len(nearest) > 1:
            continue  # skip if nearest neighbor is ambiguous

        b1_index = nearest[0][1]
        second_neighbors = [d for d in distances if abs(d[0] - d1) >= tol]
        if not second_neighbors:
            continue
        d2 = second_neighbors[0][0]

        all_pairs_info.append((d1, d2, ia, b1_index))
        d1_values.append(d1)

    if not all_pairs_info:
        print("No unique nearest neighbors found.")
        return

    # Use minimum A–D distance as threshold for filtering
    d1_min = np.min(a_d_distances)

    # Only consider A atoms whose d1 ≈ d1_min
    d2_candidates = [(d2, d1, ia, ib) for (d1, d2, ia, ib) in all_pairs_info if abs(d1 - d1_min) < tol]
    if not d2_candidates:
        print(f"No {args.element_a} atoms found with d1 ≈ d1_min ({d1_min:.6f} Å).")
        return

    # Determine min d2 and optionally filter candidates with d2 > min
    min_d2 = min(d2 for d2, _, _, _ in d2_candidates)

    if args.disable_high_d2:
        filtered_candidates = d2_candidates
    else:
        filtered_candidates = [(d2, d1, ia, ib) for (d2, d1, ia, ib) in d2_candidates if d2 > min_d2 + tol]
        if not filtered_candidates:
            print("No candidates with d2 significantly larger than the minimum.")
            return

    # Shuffle candidates for stochastic selection
    if args.random_seed is not None:
        random.seed(args.random_seed)
    random.shuffle(filtered_candidates)

    selected = []
    used_b1_indices = set()

    # Select replacements ensuring no duplicate B1 atoms are used
    for d2, d1, ia, ib in filtered_candidates:
        if ib in used_b1_indices:
            continue
        selected.append((d2, d1, ia, ib))
        used_b1_indices.add(ib)
        if len(selected) == n_replace:
            break

    if len(selected) < n_replace:
        print(f"Warning: Only {len(selected)} {args.element_a} atoms found with d1 ≈ d1_min. Replacing these.")
        n_replace = len(selected)

    # Get selected A and B1 indices
    selected_a_indices = [ia for _, _, ia, _ in selected]
    replaced_b1_indices = [ib for _, _, _, ib in selected]

    # Modify element list: replace A with C
    for ia in selected_a_indices:
        element_list[ia] = args.element_c

    # Save replacement information to CSV if requested
    if args.save_csv:
        with open(args.csv_filename, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{args.element_a} index", "x", "y", "z", f"{args.element_b} index", "x", "y", "z", "d1 (Å)", "d2 (Å)", "Replaced"])

            replaced_rows = []
            non_replaced_rows = []

            for d1, d2, ia, ib in all_pairs_info:
                row = [
                    ia, *coords_cart[ia], ib, *coords_cart[ib], f"{d1:.6f}", f"{d2:.6f}",
                    "Yes" if ia in selected_a_indices else "No"
                ]
                if ia in selected_a_indices:
                    replaced_rows.append(row)
                else:
                    non_replaced_rows.append(row)

            for row in replaced_rows + non_replaced_rows:
                writer.writerow(row)
            print(f"Replacement info saved to '{args.csv_filename}'.")

    # Sort output structure by chemical/functional priority
    replaced_b1_set = set(replaced_b1_indices)
    replaced_a_set = set(selected_a_indices)
    sorted_indices = []

    # Sort: replaced B, remaining B, replaced A, D, remaining A, others
    for i in indices_b:
        if i in replaced_b1_set:
            sorted_indices.append(i)
            comments[i] += f" => {args.element_b}3+"
    for i in indices_b:
        if i not in replaced_b1_set:
            sorted_indices.append(i)
            comments[i] += f" => {args.element_b}2+"
    for i in selected_a_indices:
        sorted_indices.append(i)
        comments[i] += f" => {args.element_c}"
    sorted_indices += [i for i in indices_d]
    sorted_indices += [i for i in indices_a if i not in replaced_a_set]
    sorted_indices += [i for i in indices if i not in indices_b and i not in indices_a and i not in indices_d]

    # Reorder structure accordingly
    sorted_coords_cart = [coords_cart[i] for i in sorted_indices]
    sorted_coords_direct = cartesian_to_fractional(np.array(sorted_coords_cart), lattice)
    sorted_comments = [comments[i] for i in sorted_indices]
    sorted_elements = [element_list[i] for i in sorted_indices]

    # Generate new element list and count
    new_elements = []
    new_counts = []

    if args.element_c == args.element_b:
        elem_list = [args.element_b, args.element_d, args.element_a]
    else:
        elem_list = [args.element_b, args.element_c, args.element_d, args.element_a]
    for elem in elem_list:
        count = sorted_elements.count(elem)
        if count > 0:
            new_elements.append(elem)
            new_counts.append(count)

    for elem in elements:
        if elem not in new_elements:
            count = sorted_elements.count(elem)
            if count > 0:
                new_elements.append(elem)
                new_counts.append(count)

    # Write modified POSCAR file in Direct coordinates
    with open(args.output_file, 'w') as f:
        f.write(f"{data['system']} - modified\n")
        f.write("1.000000\n")
        for vec in lattice:
            f.write(f"  {vec[0]:.9f}  {vec[1]:.9f}  {vec[2]:.9f}\n")
        f.write("  " + "  ".join(new_elements) + "\n")
        f.write("  " + "  ".join(map(str, new_counts)) + "\n")
        f.write("Direct\n")
        for coord, comment in zip(sorted_coords_direct, sorted_comments):
            f.write(f"  {coord[0]:.9f}  {coord[1]:.9f}  {coord[2]:.9f}  {comment}\n")

    print(f"Replaced {len(selected_a_indices)} '{args.element_a}' atoms with '{args.element_c}'.")
    print(f"New structure written to '{args.output_file}' in Direct coordinates.")

if __name__ == "__main__":
    main()
