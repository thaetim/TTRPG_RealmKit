import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from functools import lru_cache
import math

from biome_arrange_constants import *

# ===== Core Functions =====
@lru_cache(maxsize=1024)  # Cache up to 1024 unique colors
def closest_color(pixel_tuple, color_map_name):
    """
    Cached version of closest color matching.
    Uses tuple input and color map name for cache efficiency.
    """
    # Select the appropriate color map
    color_map = PGEN_COLORS if color_map_name == "pgen" else SPACEGEO_COLORS
    
    # Convert to numpy arrays for vectorized calculation
    pixel_array = np.array(pixel_tuple, dtype=np.int16)
    colors = np.array(list(color_map), dtype=np.int16)
    
    # Calculate squared distances
    diff = colors - pixel_array
    distances = np.sum(diff**2, axis=1)
    
    # Return closest color
    return list(color_map)[np.argmin(distances)]

def get_biome(pixel, color_map):
    """Map RGB pixel to biome name using cached color matching."""
    # Convert pixel to tuple for hashability (required for caching)
    pixel_tuple = tuple(pixel)
    # Determine which color map we're using
    color_map_name = "pgen" if color_map is PGEN_COLORS else "spacegeo"
    # Get closest color from cache
    closest = closest_color(pixel_tuple, color_map_name)
    return color_map[closest]

# ===== Matrix-Based Köppen Classification =====
def load_koppen_matrix(filepath):
    """Load the Köppen classification matrix from CSV."""
    import csv
    matrix = {}
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)[1:]  # PGEN biomes (skip first empty cell)
        
        for row in reader:
            spacegeo_biome = row[0]
            matrix[spacegeo_biome] = {}
            
            for i, cell in enumerate(row[1:]):
                pgen_biome = headers[i]
                matrix[spacegeo_biome][pgen_biome] = cell if cell != "-" else None
    
    return matrix

# Load the matrix at module level (will be initialized once)
KOPPEN_MATRIX = load_koppen_matrix("koppen_matrix.csv")

def parse_koppen_rule(rule_str, elev=None, lat=None):
    if not rule_str or rule_str == "-":
        return None
    
    # Context with allowed variables/functions
    context = {
        'elev': elev,
        'lat': lat,
        'sqrt': math.sqrt,
        'log': math.log
    }
    
    for option in rule_str.split("|"):
        option = option.strip()
        
        if "if" not in option:  # Default case
            return option.split()[0]
            
        koppen_class, condition = option.split("if")
        koppen_class = koppen_class.strip()
        condition = condition.replace(")", "").strip()
        
        try:
            if eval(condition, {}, context):  # Safe eval with no builtins
                return koppen_class
        except:
            continue
    
    return None

def classify_koppen(pgen_pixel, spacegeo_pixel, elevation, lat_norm):
    """Matrix-based Köppen classification with elevation/latitude rules."""
    # Skip ocean pixels
    if (tuple(spacegeo_pixel) == (76, 102, 178) or 
        tuple(pgen_pixel) == (76, 102, 178) or
        elevation < 127):
        return 'Ocean'
    
    # Get normalized elevation (0-1)
    norm_elev = elevation / 255
    abs_lat = abs(lat_norm * 90)  # Absolute latitude in degrees
    
    # Get biomes from both maps
    pgen_biome = get_biome(pgen_pixel, PGEN_COLORS)
    spacegeo_biome = get_biome(spacegeo_pixel, SPACEGEO_COLORS)
    
    # ===== Elevation Overrides =====
    if norm_elev > 0.8:  # High mountains
        if norm_elev > 0.9 or pgen_biome == 'Ice':
            return 'EF'
        return 'ET'
    
    # ===== Matrix-Based Classification =====
    # Try to find a rule in the matrix
    if spacegeo_biome in KOPPEN_MATRIX and pgen_biome in KOPPEN_MATRIX[spacegeo_biome]:
        rule = KOPPEN_MATRIX[spacegeo_biome][pgen_biome]
        if rule:
            result = parse_koppen_rule(rule, norm_elev, abs_lat)
            if result:
                return result
    
    # ===== Fallback Rules =====
    # Priority to spacegeo for specific biomes
    if spacegeo_biome in SPACEGEO_COLORS:
        if spacegeo_biome in BIOME_TO_KOPPEN:
            return BIOME_TO_KOPPEN[spacegeo_biome]
    
    # Default to pgen biome mapping
    return BIOME_TO_KOPPEN.get(pgen_biome, 'BSk')

def validate_map_dimensions(pgen, spacegeo, heightmap):
    """Validate that all maps have compatible dimensions."""
    print("Validating map dimensions...")
    if pgen.shape != spacegeo.shape:
        raise ValueError(f"Dimension mismatch: pgen {pgen.shape} vs spacegeo {spacegeo.shape}")
    if pgen.shape[:2] != heightmap.shape:
        raise ValueError(f"Heightmap dimension mismatch: {heightmap.shape} expected {pgen.shape[:2]}")
    if len(pgen.shape) != 3 or pgen.shape[2] != 3:
        raise ValueError(f"pgen must be RGB image, got shape {pgen.shape}")
    if len(spacegeo.shape) != 3 or spacegeo.shape[2] != 3:
        raise ValueError(f"spacegeo must be RGB image, got shape {spacegeo.shape}")
    print("✓ All maps have compatible dimensions")

def generate_koppen_map(pgen_path, spacegeo_path, heightmap_path, output_path, lat_range=(-90, 90)):
    """Generate Köppen map from inputs."""
    print("\nLoading maps...")
    # Load images with progress indication
    with tqdm(desc="Loading pgen map", unit="file") as pbar:
        pgen = np.array(Image.open(pgen_path)).astype(np.uint8)
        pbar.update()
    
    with tqdm(desc="Loading spacegeo map", unit="file") as pbar:
        spacegeo = np.array(Image.open(spacegeo_path).convert('RGB')).astype(np.uint8)
        pbar.update()
    
    with tqdm(desc="Loading heightmap", unit="file") as pbar:
        heightmap = np.array(Image.open(heightmap_path).convert('L')).astype(np.uint8)
        pbar.update()
    
    # Validate dimensions
    validate_map_dimensions(pgen, spacegeo, heightmap)
    
    height, width, _ = pgen.shape
    koppen_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Latitude normalization (y-axis based!)
    lat_min, lat_max = lat_range
    y_coords = np.linspace(lat_max, lat_min, height)  # y=0 is top (north)
    lat_norms = (y_coords - (lat_max + lat_min)/2) / (lat_max - lat_min)*2
    
    print("\nGenerating Köppen map...")
    # Define ocean color for comparison
    OCEAN_COLOR = np.array([76, 102, 178], dtype=np.uint8)
    
    # Process each pixel with progress bar
    for y in tqdm(range(height), desc="Processing rows", unit="row"):
        for x in range(width):
            # Skip ocean pixels
            if np.array_equal(spacegeo[y, x], OCEAN_COLOR):
                continue
                
            # Calculate effective latitude accounting for projection distortion
            raw_lat = lat_norms[y] * 90
            effective_lat_norm = raw_lat * np.cos(np.radians(abs(raw_lat))) / 90
            
            koppen_class = classify_koppen(
                pgen[y, x], 
                spacegeo[y, x], 
                heightmap[y, x],
                effective_lat_norm  # Use adjusted latitude
            )
            if koppen_class is not None:
                koppen_img[y, x] = KOPPEN_COLORS[koppen_class]
    
    print("\nSaving output...")
    with tqdm(desc="Saving Köppen map", unit="file") as pbar:
        Image.fromarray(koppen_img).save(output_path)
        pbar.update()
    
    print(f"\n✓ Köppen map saved to {output_path}")
def analyze_biome_combinations(pgen_path, spacegeo_path, output_file=None):
    """
    Analyze and report all unique pgen-spacegeo biome combinations in the maps.
    Optimized to skip ocean pixels more efficiently.
    """
    print("\nLoading maps for biome combination analysis...")
    pgen = np.array(Image.open(pgen_path)).astype(np.uint8)
    spacegeo = np.array(Image.open(spacegeo_path).convert('RGB')).astype(np.uint8)
    
    # Validate dimensions
    if pgen.shape != spacegeo.shape:
        raise ValueError("pgen and spacegeo maps must have same dimensions")
    
    height, width, _ = pgen.shape
    combinations = {}  # Using dict instead of set for better performance
    biome_counts = {}
    
    print("\nAnalyzing biome combinations...")
    total_pixels = height * width
    land_pixels = 0
    
    # Create a mask for ocean pixels (vectorized)
    ocean_mask = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            if np.array_equal(spacegeo[y, x], OCEAN_COLOR):
                ocean_mask[y, x] = True
    
    with tqdm(total=total_pixels, desc="Processing pixels") as pbar:
        for y in range(height):
            for x in range(width):
                # Skip ocean pixels using precomputed mask
                if ocean_mask[y, x]:
                    pbar.update(1)
                    continue
                    
                # Get biome names
                pgen_biome = get_biome(pgen[y, x], "pgen")
                spacegeo_biome = get_biome(spacegeo[y, x], "spacegeo")
                
                # Record combination
                combo = (pgen_biome, spacegeo_biome)
                combinations[combo] = True
                
                # Count occurrences
                biome_counts[combo] = biome_counts.get(combo, 0) + 1
                land_pixels += 1
                
                pbar.update(1)
    
    # Sort combinations by frequency (descending)
    sorted_combinations = sorted(combinations.keys(), key=lambda x: -biome_counts[x])
    
    # Prepare report
    report_lines = [
        "Unique PGen-SpaceGeo Biome Combinations Analysis",
        "==============================================",
        f"Total unique combinations found: {len(combinations)}",
        f"Total land pixels analyzed: {land_pixels} (skipped {total_pixels - land_pixels} ocean pixels)",
        "\nCombinations (sorted by frequency):",
        "PGen Biome\t\tSpaceGeo Biome\t\tCount\t%Land"
    ]
    
    for combo in sorted_combinations:
        pgen_b, spacegeo_b = combo
        count = biome_counts[combo]
        percentage = (count / land_pixels) * 100
        report_lines.append(f"{pgen_b.ljust(20)}\t{spacegeo_b.ljust(20)}\t{count}\t{percentage:.1f}%")
    
    report = "\n".join(report_lines)
    print("\n" + report)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\n✓ Analysis saved to {output_file}")
    else:
        print("\nℹ️ No output file specified - results only printed to console")
    
    return combinations, biome_counts

def analyze_koppen_distributions(koppen_path, heightmap_path, lat_range=(-90, 90), output_file=None):
    """
    Analyze distributions of Köppen classes by altitude and latitude.
    Optimized to skip ocean pixels and use vectorized operations where possible.
    """
    print("\nLoading data for distribution analysis...")
    
    # Load maps
    with tqdm(desc="Loading maps", total=2) as pbar:
        koppen_img = np.array(Image.open(koppen_path))
        pbar.update()
        
        heightmap = np.array(Image.open(heightmap_path).convert('L')) / 255.0
        pbar.update()
    
    # Validate dimensions
    if koppen_img.shape[:2] != heightmap.shape:
        raise ValueError("Köppen map and heightmap must have same dimensions")
    
    height, width = heightmap.shape
    
    # Use dictionary of numpy arrays for better statistics performance
    stats = {
        'by_class': {},
        'altitude_bins': {},
        'latitude_bins': {}
    }
    
    # Prepare reverse color mapping
    OCEAN_COLOR_RGB = np.array(KOPPEN_COLORS['Ocean'])
    
    # Latitude bins
    lat_min, lat_max = lat_range
    y_coords = np.linspace(lat_max, lat_min, height)
    
    # Bin definitions
    ALTITUDE_BINS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    LATITUDE_BINS = [-90, -60, -30, 0, 30, 60, 90]
    
    print("\nAnalyzing distributions...")
    
    # Create ocean mask once (vectorized where possible)
    ocean_mask = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            ocean_mask[y, x] = np.array_equal(koppen_img[y, x], OCEAN_COLOR_RGB)
    
    # Initialize data structures for each Köppen class
    koppen_classes = set()
    total_land_pixels = 0
    
    # First pass: identify all Köppen classes and count land pixels
    for y in range(height):
        for x in range(width):
            if not ocean_mask[y, x]:
                # Get Köppen class
                pixel_tuple = tuple(koppen_img[y, x])
                koppen_class = COLOR_TO_KOPPEN.get(pixel_tuple, 'Unknown')
                koppen_classes.add(koppen_class)
                total_land_pixels += 1
    
    # Initialize statistics arrays for each Köppen class
    for koppen_class in koppen_classes:
        stats['by_class'][koppen_class] = {
            'count': 0,
            'elevation_sum': 0,
            'lat_sum': 0,
            'min_alt': 1.0,
            'max_alt': 0.0,
            'pixels': []  # Store pixels for later analysis
        }
    
    # Process each pixel with progress bar (now with optimized ocean skipping)
    with tqdm(total=height*width, desc="Processing pixels") as pbar:
        for y in range(height):
            current_lat = y_coords[y]
            lat_bin = np.digitize(current_lat, LATITUDE_BINS) - 1  # 0-based
            
            for x in range(width):
                # Skip ocean pixels using precomputed mask
                if ocean_mask[y, x]:
                    pbar.update(1)
                    continue
                    
                # Get Köppen class
                pixel_tuple = tuple(koppen_img[y, x])
                koppen_class = COLOR_TO_KOPPEN.get(pixel_tuple, 'Unknown')
                
                # Get elevation (0-1)
                elevation = heightmap[y, x]
                alt_bin = np.digitize(elevation, ALTITUDE_BINS) - 1  # 0-based
                
                # Update class statistics
                stats['by_class'][koppen_class]['count'] += 1
                stats['by_class'][koppen_class]['elevation_sum'] += elevation
                stats['by_class'][koppen_class]['lat_sum'] += current_lat
                stats['by_class'][koppen_class]['min_alt'] = min(
                    stats['by_class'][koppen_class]['min_alt'], elevation)
                stats['by_class'][koppen_class]['max_alt'] = max(
                    stats['by_class'][koppen_class]['max_alt'], elevation)
                
                # Add pixel to list for possible later analysis
                stats['by_class'][koppen_class]['pixels'].append((x, y, elevation, current_lat))
                
                # Update altitude bin stats
                if alt_bin not in stats['altitude_bins']:
                    stats['altitude_bins'][alt_bin] = {}
                stats['altitude_bins'][alt_bin][koppen_class] = (
                    stats['altitude_bins'][alt_bin].get(koppen_class, 0) + 1)
                
                # Update latitude bin stats
                if lat_bin not in stats['latitude_bins']:
                    stats['latitude_bins'][lat_bin] = {}
                stats['latitude_bins'][lat_bin][koppen_class] = (
                    stats['latitude_bins'][lat_bin].get(koppen_class, 0) + 1)
                
                pbar.update(1)
    
    # Calculate averages
    for koppen_class in stats['by_class']:
        count = stats['by_class'][koppen_class]['count']
        if count > 0:
            stats['by_class'][koppen_class]['mean_alt'] = (
                stats['by_class'][koppen_class]['elevation_sum'] / count)
            stats['by_class'][koppen_class]['mean_lat'] = (
                stats['by_class'][koppen_class]['lat_sum'] / count)
        else:
            stats['by_class'][koppen_class]['mean_alt'] = 0
            stats['by_class'][koppen_class]['mean_lat'] = 0
        
        # Remove raw pixel data to save memory after calculations
        del stats['by_class'][koppen_class]['pixels']
    
    # Generate report
    report_lines = [
        "Köppen Climate Distribution Analysis",
        "=================================",
        f"Total land pixels analyzed: {total_land_pixels}",
        "\nOverall Statistics by Climate Class:",
        "Class\tCount\t%Total\tMeanAlt\tMinAlt\tMaxAlt\tMeanLat"
    ]
    
    for koppen_class in sorted(stats['by_class'], 
                             key=lambda x: -stats['by_class'][x]['count']):
        data = stats['by_class'][koppen_class]
        pct = (data['count'] / total_land_pixels) * 100
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
        bin_max = ALTITUDE_BINS[bin_idx + 1] if bin_idx + 1 < len(ALTITUDE_BINS) else 1.0
        classes = sorted(stats['altitude_bins'][bin_idx].items(),
                       key=lambda x: -x[1])
        class_str = ", ".join(f"{k}:{v}" for k,v in classes)
        report_lines.append(
            f"{bin_idx}\t{bin_min:.1f}-{bin_max:.1f}\t{class_str}")
    
    # Latitude distribution
    report_lines.extend([
        "\n\nLatitude Distribution (by latitude zone):",
        "Bin\tRange\t\tClasses (count)..."
    ])
    for bin_idx in sorted(stats['latitude_bins']):
        bin_min = LATITUDE_BINS[bin_idx]
        bin_max = LATITUDE_BINS[bin_idx + 1] if bin_idx + 1 < len(LATITUDE_BINS) else 90
        classes = sorted(stats['latitude_bins'][bin_idx].items(),
                       key=lambda x: -x[1])
        class_str = ", ".join(f"{k}:{v}" for k,v in classes)
        report_lines.append(
            f"{bin_idx}\t{bin_min:>3}°-{bin_max:>3}°\t{class_str}")
    
    report = "\n".join(report_lines)
    print("\n" + report)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\n✓ Analysis saved to {output_file}")
    
    return stats

# ===== CLI =====
if __name__ == "__main__":
    # Default file paths
    MAIN_WDIR = Path(r"D:\DND\Realistic DND World Gen\renders")
    DEFAULT_PGEN = MAIN_WDIR / "climate.bmp"
    DEFAULT_SPACEGEO = MAIN_WDIR / "canvas tuned.png"
    DEFAULT_HEIGHTMAP = MAIN_WDIR / "greyscale tuned.bmp"
    DEFAULT_OUTPUT = MAIN_WDIR / "koppen tuned.bmp"
    FPATH_AN_INPUT_BIOMECOMBS = MAIN_WDIR / "analysis-biomecombs.txt"
    FPATH_AN_RESULT_DISTRIBUTIONS = MAIN_WDIR / "analysis-distributions.txt"

    parser = argparse.ArgumentParser(
        description="Generate a Köppen map for a fantasy planet based on inputs from Planet Generator and Space Geometrian.",
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
    parser.add_argument("--dist-output", default=FPATH_AN_RESULT_DISTRIBUTIONS,
                      help="Output file for distribution analysis")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_biome_combinations(
            args.pgen, args.spacegeo, args.analysis_output
        )
    elif args.dist:
        analyze_koppen_distributions(
            args.output, args.heightmap, args.lat, args.dist_output
        )
    else:
        generate_koppen_map(
            args.pgen, args.spacegeo, args.heightmap, 
            args.output, args.lat
        )