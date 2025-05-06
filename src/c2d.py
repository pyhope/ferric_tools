#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yihang Peng
# Date: 05/05/2025

import numpy as np
import argparse

def read_poscar(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    system_name = lines[0].strip()
    scale = float(lines[1].strip())
    lattice_vectors = np.array([list(map(float, lines[i].split())) for i in range(2, 5)])
    lattice_vectors *= scale

    element_names = lines[5].split()
    element_counts = list(map(int, lines[6].split()))
    coord_type = lines[7].strip().lower()
    num_atoms = sum(element_counts)

    coord_lines = lines[8:8 + num_atoms]
    coords = []
    comments = []

    for line in coord_lines:
        if '!' in line:
            coord_part, comment_part = line.split('!', 1)
            comment = '! ' + comment_part.strip()
        else:
            coord_part = line
            comment = ''
        values = list(map(float, coord_part.strip().split()))
        coords.append(values)
        comments.append(comment)

    return {
        'system': system_name,
        'lattice': lattice_vectors,
        'elements': element_names,
        'counts': element_counts,
        'coord_type': coord_type,
        'coordinates': np.array(coords),
        'comments': comments
    }

def cartesian_to_fractional(cart_coords, lattice):
    inv_lattice = np.linalg.inv(lattice.T)
    return np.dot(cart_coords, inv_lattice)

def fractional_to_cartesian(frac_coords, lattice):
    return np.dot(frac_coords, lattice.T)

def write_poscar(data, new_coords, new_type, output_filename):
    with open(output_filename, 'w') as f:
        f.write(f"{data['system']}\n")
        f.write("1.000000\n")
        for vec in data['lattice']:
            f.write(f"  {vec[0]:.9f}  {vec[1]:.9f}  {vec[2]:.9f}\n")
        f.write("  " + "  ".join(data['elements']) + "\n")
        f.write("  " + " ".join(map(str, data['counts'])) + "\n")
        f.write(f"{new_type.capitalize()}\n")
        for coord, comment in zip(new_coords, data['comments']):
            coord_str = f"  {coord[0]:.9f}  {coord[1]:.9f}  {coord[2]:.9f}"
            f.write(f"{coord_str:<45}{comment}\n")

def main():
    parser = argparse.ArgumentParser(description='Convert between Cartesian and Direct coordinates in a POSCAR-like file.')
    parser.add_argument("--input_file", "-i", type=str, default="POSCAR", help="Input POSCAR file (default: POSCAR)")
    parser.add_argument("--output_file", "-o", type=str, default="conf.vasp", help="Output converted file (default: conf.vasp)")
    args = parser.parse_args()

    data = read_poscar(args.input_file)

    if data['coord_type'] == 'cartesian':
        converted_coords = cartesian_to_fractional(data['coordinates'], data['lattice'])
        new_type = 'Direct'
    elif data['coord_type'] == 'direct':
        converted_coords = fractional_to_cartesian(data['coordinates'], data['lattice'])
        new_type = 'Cartesian'
    else:
        raise ValueError("Unsupported coordinate type: must be 'Cartesian' or 'Direct'.")

    write_poscar(data, converted_coords, new_type, args.output_file)
    print(f"Conversion completed. Output written to '{args.output_file}' in {new_type} format.")

if __name__ == '__main__':
    main()
