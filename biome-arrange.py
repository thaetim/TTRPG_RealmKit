import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numba
import csv

from biome_arrange_constants import *

# ===== Constants and Precomputations =====

# Precompute color mappings as numpy arrays for faster access
PGEN_COLORS_ARRAY = np.array(list(PGEN_COLORS.keys()), dtype=np.uint8)
SPACEGEO_COLORS_ARRAY = np.array(list(SPACEGEO_COLORS.keys()), dtype=np.uint8)
OCEAN_COLOR = np.array([76, 102, 178], dtype=np.uint8)


# ===== Numba-Compatible Data Structures =====

# Convert color mappings to numpy arrays
PGEN_COLORS_KEYS = np.array(list(PGEN_COLORS.keys()), dtype=np.uint8)
# String array for biome names
PGEN_COLORS_VALUES = np.array(list(PGEN_COLORS.values()), dtype='U20')
SPACEGEO_COLORS_KEYS = np.array(list(SPACEGEO_COLORS.keys()), dtype=np.uint8)
SPACEGEO_COLORS_VALUES = np.array(list(SPACEGEO_COLORS.values()), dtype='U20')

# Convert Koppen matrix to Numba-compatible format


def prepare_koppen_matrix(matrix):
    """Convert the Koppen matrix to Numba-compatible arrays."""
    spacegeo_biomes = list(matrix.keys())
    pgen_biomes = list(matrix[spacegeo_biomes[0]].keys())

    # Create 2D array of rules
    rules = np.empty((len(spacegeo_biomes), len(pgen_biomes)), dtype='U50')
    for i, sg_biome in enumerate(spacegeo_biomes):
        for j, pg_biome in enumerate(pgen_biomes):
            rules[i, j] = matrix[sg_biome][pg_biome] if matrix[sg_biome][pg_biome] else '-'

    return spacegeo_biomes, pgen_biomes, rules

# ===== Numba-Optimized Functions =====


@numba.njit
def find_closest_color(pixel, color_array):
    """Numba-optimized closest color finder."""
    min_dist = np.inf
    closest_idx = 0
    for i in range(color_array.shape[0]):
        dist = 0
        for j in range(3):
            diff = int(pixel[j]) - int(color_array[i, j])
            dist += diff * diff
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    return closest_idx


# ===== Optimized Core Functions =====


def load_koppen_matrix(filepath):
    """Load the Köppen classification matrix with caching."""
    matrix = {}
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)[1:]  # PGEN biomes

        for row in reader:
            spacegeo_biome = row[0]
            matrix[spacegeo_biome] = dict(zip(headers, row[1:]))

    return matrix


# ===== Optimized Numba Functions =====

@numba.njit
def find_biome_index(biome_name, biome_list):
    """Find index of biome in list."""
    for i in range(len(biome_list)):
        if biome_list[i] == biome_name:
            return i
    return -1


# ===== Numba-Compatible Data Structures =====

# Convert Koppen colors to array format
KOPPEN_CLASSES = np.array(list(KOPPEN_TO_COLOR.keys()), dtype='U10')
KOPPEN_COLORS_ARRAY = np.array(list(KOPPEN_TO_COLOR.values()), dtype=np.uint8)


@numba.njit
def get_koppen_color(koppen_class, classes, colors):
    """Numba-compatible color lookup."""
    for i in range(len(classes)):
        if classes[i] == koppen_class:
            return colors[i]
    return colors[0]  # Default color


@numba.njit(parallel=True)
def classify_koppen_batch(pgen_pixels, spacegeo_pixels, heightmap, lat_norms,
                          spacegeo_biomes, pgen_biomes, koppen_rules,
                          koppen_classes, koppen_colors):
    """Batch processing of pixels with Numba."""
    height, width = heightmap.shape
    output = np.zeros((height, width, 3), dtype=np.uint8)

    # Prepare ocean color check
    ocean_color = np.array([76, 102, 178], dtype=np.uint8)

    for y in numba.prange(height):
        for x in range(width):
            # Skip ocean pixels
            if (spacegeo_pixels[y, x, 0] == ocean_color[0] and
                spacegeo_pixels[y, x, 1] == ocean_color[1] and
                    spacegeo_pixels[y, x, 2] == ocean_color[2]):
                continue

            # Get elevation and latitude
            elevation = heightmap[y, x]
            norm_elev = elevation / 255.0

            # High elevation overrides
            if norm_elev > 0.8:
                if norm_elev > 0.9:
                    output[y, x] = get_koppen_color(
                        'EF', koppen_classes, koppen_colors)
                    continue
                output[y, x] = get_koppen_color(
                    'ET', koppen_classes, koppen_colors)
                continue

            # Get biome indices
            pgen_idx = find_closest_color(pgen_pixels[y, x], PGEN_COLORS_KEYS)
            spacegeo_idx = find_closest_color(
                spacegeo_pixels[y, x], SPACEGEO_COLORS_KEYS)

            # Get biome names
            pgen_biome = PGEN_COLORS_VALUES[pgen_idx]
            spacegeo_biome = SPACEGEO_COLORS_VALUES[spacegeo_idx]

            # Find matrix indices
            sg_idx = find_biome_index(spacegeo_biome, spacegeo_biomes)
            pg_idx = find_biome_index(pgen_biome, pgen_biomes)

            # Default class
            koppen_class = 'BSk'

            # Matrix lookup
            if sg_idx >= 0 and pg_idx >= 0:
                rule = koppen_rules[sg_idx, pg_idx]
                if rule != '-':
                    # Simple rule parsing (first option before | or ()
                    parts = rule.split('|')
                    if len(parts) > 0:
                        koppen_class = parts[0].split('(')[0].strip()

            # Assign color
            output[y, x] = get_koppen_color(
                koppen_class, koppen_classes, koppen_colors)

    return output


def generate_koppen_map(pgen_path, spacegeo_path, heightmap_path, output_path, lat_range=(-90, 90)):
    """Optimized Köppen map generation."""
    print("\nLoading maps...")
    pgen = np.array(Image.open(pgen_path))
    spacegeo = np.array(Image.open(spacegeo_path).convert('RGB'))
    heightmap = np.array(Image.open(heightmap_path).convert('L'))

    # Validate dimensions
    if pgen.shape != spacegeo.shape or pgen.shape[:2] != heightmap.shape:
        raise ValueError("Input maps must have compatible dimensions")

    # Precompute latitude norms
    height = pgen.shape[0]
    lat_min, lat_max = lat_range
    y_coords = np.linspace(lat_max, lat_min, height)
    lat_norms = (y_coords - (lat_max + lat_min)/2) / (lat_max - lat_min)*2

    print("\nPreparing data structures...")
    # Prepare Koppen matrix for Numba
    spacegeo_biomes, pgen_biomes, koppen_rules = prepare_koppen_matrix(
        KOPPEN_MATRIX)

    print("\nGenerating Köppen map...")
    # Process in batches using Numba
    koppen_img = classify_koppen_batch(
        pgen, spacegeo, heightmap, lat_norms,
        np.array(spacegeo_biomes), np.array(pgen_biomes),
        koppen_rules, KOPPEN_CLASSES, KOPPEN_COLORS_ARRAY
    )

    print("\nSaving output...")
    Image.fromarray(koppen_img).save(output_path)
    print(f"\n✓ Köppen map saved to {output_path}")


# ===== Optimized Analysis Functions =====


def analyze_biome_combinations(pgen_path, spacegeo_path, output_file=None):
    """Vectorized biome combination analysis."""
    print("\nLoading maps...")
    pgen = np.array(Image.open(pgen_path))
    spacegeo = np.array(Image.open(spacegeo_path).convert('RGB'))

    if pgen.shape != spacegeo.shape:
        raise ValueError("Maps must have same dimensions")

    # Create ocean mask (vectorized)
    ocean_mask = np.all(spacegeo == OCEAN_COLOR, axis=-1)
    land_mask = ~ocean_mask

    # Get biome indices (vectorized)
    pgen_biomes = np.empty(pgen.shape[:2], dtype=object)
    spacegeo_biomes = np.empty(spacegeo.shape[:2], dtype=object)

    print("\nFinding biome combinations...")
    for y in tqdm(range(pgen.shape[0])):
        for x in range(pgen.shape[1]):
            if land_mask[y, x]:
                pgen_idx = find_closest_color(pgen[y, x], PGEN_COLORS_ARRAY)
                spacegeo_idx = find_closest_color(
                    spacegeo[y, x], SPACEGEO_COLORS_ARRAY)
                pgen_biomes[y, x] = list(PGEN_COLORS.values())[pgen_idx]
                spacegeo_biomes[y, x] = list(SPACEGEO_COLORS.values())[
                    spacegeo_idx]

    # Count combinations
    unique, counts = np.unique(
        np.stack((pgen_biomes[land_mask],
                 spacegeo_biomes[land_mask]), axis=-1),
        axis=0,
        return_counts=True
    )

    # Generate report
    report_lines = [
        "Unique PGen-SpaceGeo Biome Combinations Analysis",
        "==============================================",
        f"Total unique combinations found: {len(unique)}",
        f"Total land pixels analyzed: {land_mask.sum()}",
        "\nCombinations (sorted by frequency):",
        "PGen Biome\t\tSpaceGeo Biome\t\tCount\t%Land"
    ]

    sorted_indices = np.argsort(-counts)
    for idx in sorted_indices:
        pgen_b, spacegeo_b = unique[idx]
        count = counts[idx]
        percentage = (count / land_mask.sum()) * 100
        report_lines.append(
            f"{pgen_b.ljust(20)}\t{spacegeo_b.ljust(20)}\t{count}\t{percentage:.1f}%")

    report = "\n".join(report_lines)
    print("\n" + report)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\n✓ Analysis saved to {output_file}")


def analyze_koppen_distributions(koppen_path, heightmap_path, lat_range=(-90, 90), output_file=None):
    """Optimized Köppen distribution analysis by altitude/latitude."""
    print("\nLoading data for distribution analysis...")

    # Load maps as numpy arrays
    koppen_img = np.array(Image.open(koppen_path))
    heightmap = np.array(Image.open(heightmap_path).convert('L')) / 255.0

    # Validate dimensions
    if koppen_img.shape[:2] != heightmap.shape:
        raise ValueError("Köppen map and heightmap must have same dimensions")

    height, width = heightmap.shape
    total_pixels = height * width

    # Create ocean mask (vectorized)
    ocean_mask = np.all(koppen_img == OCEAN_COLOR, axis=-1)
    land_mask = ~ocean_mask
    land_pixels = np.sum(land_mask)

    # Precompute latitude values
    lat_min, lat_max = lat_range
    y_coords = np.linspace(lat_max, lat_min, height)
    latitudes = np.tile(y_coords[:, np.newaxis], (1, width))

    # Bin definitions
    ALTITUDE_BINS = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    LATITUDE_BINS = np.array([-90, -60, -30, 0, 30, 60, 90])

    # Initialize data structures
    stats = {
        'by_class': {},
        'altitude_bins': {i: {} for i in range(len(ALTITUDE_BINS)-1)},
        'latitude_bins': {i: {} for i in range(len(LATITUDE_BINS)-1)}
    }

    print("\nProcessing pixels...")

    # Get all unique Köppen classes present in the image
    unique_colors = np.unique(koppen_img[land_mask].reshape(-1, 3), axis=0)
    koppen_classes = set()

    for color in unique_colors:
        color_tuple = tuple(color)
        if color_tuple in COLOR_TO_KOPPEN:
            koppen_classes.add(COLOR_TO_KOPPEN[color_tuple])

    # Initialize statistics for each class
    for koppen_class in koppen_classes:
        stats['by_class'][koppen_class] = {
            'count': 0,
            'elevation_sum': 0.0,
            'lat_sum': 0.0,
            'min_alt': 1.0,
            'max_alt': 0.0
        }

    # Process land pixels only
    land_coords = np.argwhere(land_mask)
    land_elevations = heightmap[land_mask]
    land_latitudes = latitudes[land_mask]
    land_colors = koppen_img[land_mask]

    for idx in tqdm(range(len(land_coords)), desc="Analyzing distributions"):
        y, x = land_coords[idx]
        elevation = land_elevations[idx]
        lat = land_latitudes[idx]
        color = land_colors[idx]
        color_tuple = tuple(color)

        if color_tuple not in COLOR_TO_KOPPEN:
            continue

        koppen_class = COLOR_TO_KOPPEN[color_tuple]
        class_stats = stats['by_class'][koppen_class]

        # Update class statistics
        class_stats['count'] += 1
        class_stats['elevation_sum'] += elevation
        class_stats['lat_sum'] += lat
        class_stats['min_alt'] = min(class_stats['min_alt'], elevation)
        class_stats['max_alt'] = max(class_stats['max_alt'], elevation)

        # Update altitude bins
        alt_bin = np.digitize(elevation, ALTITUDE_BINS) - 1
        if alt_bin >= 0 and alt_bin < len(ALTITUDE_BINS)-1:
            stats['altitude_bins'][alt_bin][koppen_class] = stats['altitude_bins'][alt_bin].get(
                koppen_class, 0) + 1

        # Update latitude bins
        lat_bin = np.digitize(lat, LATITUDE_BINS) - 1
        if lat_bin >= 0 and lat_bin < len(LATITUDE_BINS)-1:
            stats['latitude_bins'][lat_bin][koppen_class] = stats['latitude_bins'][lat_bin].get(
                koppen_class, 0) + 1

    # Calculate averages
    for koppen_class in stats['by_class']:
        count = stats['by_class'][koppen_class]['count']
        if count > 0:
            stats['by_class'][koppen_class]['mean_alt'] = stats['by_class'][koppen_class]['elevation_sum'] / count
            stats['by_class'][koppen_class]['mean_lat'] = stats['by_class'][koppen_class]['lat_sum'] / count

    # Generate report
    report_lines = [
        "Köppen Climate Distribution Analysis",
        "=================================",
        f"Total land pixels analyzed: {land_pixels}",
        "\nOverall Statistics by Climate Class:",
        "Class\tCount\t%Total\tMeanAlt\tMinAlt\tMaxAlt\tMeanLat"
    ]

    # Sort classes by count (descending)
    sorted_classes = sorted(
        stats['by_class'].items(), key=lambda x: -x[1]['count'])

    for koppen_class, data in sorted_classes:
        pct = (data['count'] / land_pixels) * 100
        report_lines.append(
            f"{koppen_class}\t{data['count']}\t{pct:.1f}%\t"
            f"{data['mean_alt']:.3f}\t{data['min_alt']:.3f}\t"
            f"{data['max_alt']:.3f}\t{data['mean_lat']:.1f}°")

    # Altitude distribution
    report_lines.extend([
        "\n\nAltitude Distribution (by elevation bin):",
        "Bin\tRange\tClasses (count)..."
    ])

    for bin_idx in sorted(stats['altitude_bins']):
        bin_min = ALTITUDE_BINS[bin_idx]
        bin_max = ALTITUDE_BINS[bin_idx + 1] if bin_idx + \
            1 < len(ALTITUDE_BINS) else 1.0
        classes = sorted(stats['altitude_bins']
                         [bin_idx].items(), key=lambda x: -x[1])
        class_str = ", ".join(f"{k}:{v}" for k, v in classes)
        report_lines.append(
            f"{bin_idx}\t{bin_min:.1f}-{bin_max:.1f}\t{class_str}")

    # Latitude distribution
    report_lines.extend([
        "\n\nLatitude Distribution (by latitude zone):",
        "Bin\tRange\t\tClasses (count)..."
    ])

    for bin_idx in sorted(stats['latitude_bins']):
        bin_min = LATITUDE_BINS[bin_idx]
        bin_max = LATITUDE_BINS[bin_idx + 1] if bin_idx + \
            1 < len(LATITUDE_BINS) else 90
        classes = sorted(stats['latitude_bins']
                         [bin_idx].items(), key=lambda x: -x[1])
        class_str = ", ".join(f"{k}:{v}" for k, v in classes)
        report_lines.append(
            f"{bin_idx}\t{bin_min:>3}°-{bin_max:>3}°\t{class_str}")

    report = "\n".join(report_lines)
    print("\n" + report)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\n✓ Analysis saved to {output_file}")

    return stats


# ===== Main CLI =====
if __name__ == "__main__":
    # Default file paths
    MAIN_WDIR = Path(r"D:\DND\Realistic DND World Gen")
    DEFAULT_PGEN = MAIN_WDIR / "renders" / "climate.bmp"
    DEFAULT_SPACEGEO = MAIN_WDIR / "renders" / "canvas tuned.png"
    DEFAULT_HEIGHTMAP = MAIN_WDIR / "renders" / "greyscale tuned.bmp"
    DEFAULT_OUTPUT = MAIN_WDIR / "climate" / "koppen tuned.bmp"
    FPATH_AN_INPUT_BIOMECOMBS = MAIN_WDIR / "climate" / "analysis-biomecombs.txt"

    # Load the matrix once
    KOPPEN_MATRIX = load_koppen_matrix(
        MAIN_WDIR / "climate" / "koppen_matrix.csv")

    parser = argparse.ArgumentParser(
        description="Generate a Köppen map for a fantasy planet with optimized performance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pgen", default=DEFAULT_PGEN,
                        help="pgen biome map (RGB)")
    parser.add_argument("--spacegeo", default=DEFAULT_SPACEGEO,
                        help="spacegeo biome map (RGB)")
    parser.add_argument("--heightmap", default=DEFAULT_HEIGHTMAP,
                        help="Heightmap (grayscale)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output Köppen map")
    parser.add_argument("--lat", nargs=2, type=float, default=[-90, 90],
                        help="Latitude range (south to north)")
    parser.add_argument("--analyze", action='store_true',
                        help="Analyze biome combinations instead of generating Köppen map")
    parser.add_argument("--analysis-output", default=FPATH_AN_INPUT_BIOMECOMBS,
                        help="Output file for biome combination analysis")
    parser.add_argument("--dist", action='store_true',
                        help="Analyze Köppen distributions by altitude/latitude")
    parser.add_argument("--dist-output",
                        help="Output file for distribution analysis")

    args = parser.parse_args()

    if args.dist:
        analyze_koppen_distributions(
            args.output, args.heightmap, args.lat, args.dist_output
        )
    elif args.analyze:
        analyze_biome_combinations(
            args.pgen, args.spacegeo, args.analysis_output
        )
    else:
        generate_koppen_map(
            args.pgen, args.spacegeo, args.heightmap,
            args.output, args.lat
        )
