import argparse
import numpy as np
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import os
import re
from biome_arrange_constants import KOPPEN_COLORS

# Constants for working directory and basic file paths
WORKING_DIR = Path("D:/DND/Realistic DND World Gen/earth")  # Proper forward slashes to avoid escape sequence issues
FPATH_KOPPEN = WORKING_DIR / "Koppen-Geiger_Map_v2_World_1991–2020.svg.png"
FPATH_ELEVATION = WORKING_DIR / "srtm_ramp2.world.5400x2700.jpg"
FPATH_OUTPUT = WORKING_DIR / "climate_palettes.json"

# Month number to abbreviation mapping
MONTH_ABBR = {
    "01": "jan", "02": "feb", "03": "mar", "04": "apr", 
    "05": "may", "06": "jun", "07": "jul", "08": "aug",
    "09": "sep", "10": "oct", "11": "nov", "12": "dec"
}

# Dynamically generate satellite image paths
FPATH_SATELLITE = {}
satellite_pattern = re.compile(r"world\.(\d{4})(\d{2})\.3x5400x2700\.jpg")

# Find all matching satellite files in the working directory
if WORKING_DIR.exists():
    for file in os.listdir(WORKING_DIR):
        match = satellite_pattern.match(file)
        if match:
            year, month = match.groups()
            if month in MONTH_ABBR:
                FPATH_SATELLITE[MONTH_ABBR[month]] = WORKING_DIR / file

# Default to an empty list if no files are found
DEFAULT_MONTHLY_MAPS = [f"{month}:{path}" for month, path in FPATH_SATELLITE.items()]

# Define month mapping
MONTH_MAPPING = {
    'jan': 'January',
    'feb': 'February',
    'mar': 'March',
    'apr': 'April',
    'may': 'May',
    'jun': 'June',
    'jul': 'July',
    'aug': 'August',
    'sep': 'September',
    'oct': 'October',
    'nov': 'November',
    'dec': 'December'
}

# Season mapping with hemispheric variations
SEASON_MAPPING = {
    # Northern Hemisphere seasons
    'north': {
        'winter': ['dec', 'jan', 'feb'],
        'spring': ['mar', 'apr', 'may'],
        'summer': ['jun', 'jul', 'aug'],
        'fall': ['sep', 'oct', 'nov']
    },
    # Southern Hemisphere seasons (opposite)
    'south': {
        'summer': ['dec', 'jan', 'feb'],
        'fall': ['mar', 'apr', 'may'],
        'winter': ['jun', 'jul', 'aug'],
        'spring': ['sep', 'oct', 'nov']
    },
    # Tropical regions (less seasonal variation)
    'tropical': {
        'dry': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'wet': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    }
}

def get_hemisphere(y, height):
    """Determine if a pixel is in the northern or southern hemisphere"""
    equator = height / 2
    return 'north' if y < equator else 'south'

def is_tropical_climate(zone):
    """Check if a climate zone is tropical (has minimal seasonal variation)"""
    return zone[0] == 'A'  # All tropical zones start with A in Koppen-Geiger

def calibrate_koppen_colors(koppen_map, sample_points=10000):
    """Helper function to find actual colors in your map"""
    height, width = koppen_map.shape[:2]
    unique_colors = {}
    
    print("Sampling colors from map for calibration...")
    for _ in range(sample_points):
        y = np.random.randint(0, height)
        x = np.random.randint(0, width)
        color = tuple(koppen_map[y, x][:3])
        
        # Skip black boundaries
        if np.linalg.norm(np.array(color) - np.array([0, 0, 0])) < 10:
            continue
            
        if color not in unique_colors:
            unique_colors[color] = 0
        unique_colors[color] += 1
    
    # Sort by frequency
    sorted_colors = sorted(unique_colors.items(), key=lambda x: -x[1])
    
    print("\nMost common non-boundary colors in your map:")
    for color, count in sorted_colors[:20]:  # Top 20 colors
        print(f"RGB {color}: {count} pixels (hex: #{color[0]:02x}{color[1]:02x}{color[2]:02x})")
    
    return sorted_colors

def color_distance(c1, c2):
    """Calculate perceptual color distance with better weighting"""
    c1 = np.array(c1)
    c2 = np.array(c2)
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return np.sqrt(2 * dr**2 + 4 * dg**2 + 3 * db**2)

def find_koppen_zones(koppen_map, tolerance=4):
    """Improved zone detection that handles interior pixels better"""
    print("Finding Koppen-Geiger zones in map...")
    height, width = koppen_map.shape[:2]
    
    rgb_to_koppen = {tuple(map(int, color)): code for code, color in KOPPEN_COLORS.items()}
    zone_pixels = defaultdict(list)
    black = (0, 0, 0)
    target_colors = [(np.array(color), code) for code, color in KOPPEN_COLORS.items()]
    exclude_colors = {(0, 0, 0), (255, 255, 255), (102, 102, 102)}
    
    with tqdm(total=height*width) as pbar:
        for y in range(height):
            for x in range(width):
                pixel = tuple(koppen_map[y, x][:3])
                
                if pixel in exclude_colors:
                    pbar.update(1)
                    continue
                
                min_dist = float('inf')
                closest_zone = None
                
                for zone_code, zone_color in KOPPEN_COLORS.items():
                    dist = color_distance(pixel, zone_color)
                    if dist < min_dist and dist < tolerance:
                        min_dist = dist
                        closest_zone = zone_code
                
                if closest_zone:
                    zone_pixels[closest_zone].append((x, y))
                
                pbar.update(1)
    
    min_pixels = 50
    filtered_zones = {zone: pixels for zone, pixels in zone_pixels.items() 
                     if len(pixels) >= min_pixels}
    
    print("\nZone detection results:")
    for zone, pixels in sorted(filtered_zones.items()):
        print(f"  {zone}: {len(pixels)} pixels")
    
    return filtered_zones

def debug_visualize_zones(koppen_map, zone_pixels, output_path="zone_debug.png"):
    """Improved debug visualization"""
    debug_img = koppen_map.copy()
    zone_colors = KOPPEN_COLORS
    
    for zone, pixels in zone_pixels.items():
        color = zone_colors.get(zone, (255, 255, 0))
        for x, y in pixels:
            debug_img[y, x, :3] = color
    
    comparison = Image.new('RGB', (koppen_map.shape[1]*2, koppen_map.shape[0]))
    comparison.paste(Image.fromarray(koppen_map), (0, 0))
    comparison.paste(Image.fromarray(debug_img), (koppen_map.shape[1], 0))
    comparison.save(output_path)
    print(f"Saved comparison visualization to {output_path}")

def resize_map(src_map, target_size):
    """Resize a map to match the target size, preserving aspect ratio."""
    src_width, src_height = src_map.size
    target_width, target_height = target_size
    ratio = min(target_width / src_width, target_height / src_height)
    new_width = int(src_width * ratio)
    new_height = int(src_height * ratio)
    resized = src_map.resize((new_width, new_height), Image.LANCZOS)
    result = Image.new("RGB", target_size, (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    result.paste(resized, (paste_x, paste_y))
    return np.array(result)

def extract_seasonal_colors(koppen_map, monthly_maps, elevations, zone_pixels, sample_size=500):
    """
    Extract representative colors for each Koppen zone from the monthly maps,
    accounting for hemispheric seasons.
    """
    koppen_height, koppen_width = koppen_map.shape[:2]
    first_month = next(iter(monthly_maps.values()))
    monthly_height, monthly_width = first_month.shape[:2]
    x_scale = monthly_width / koppen_width
    y_scale = monthly_height / koppen_height
    
    results = {}
    
    for zone, pixels in zone_pixels.items():
        print(f"\nProcessing zone: {zone}")
        
        if len(pixels) > sample_size:
            sampled_pixels = np.random.choice(len(pixels), sample_size, replace=False)
            sampled_pixels = [pixels[i] for i in sampled_pixels]
        else:
            sampled_pixels = pixels
        
        zone_results = {}
        
        # Determine if this is a tropical zone (minimal seasonal variation)
        tropical = is_tropical_climate(zone)
        
        for month, monthly_map in monthly_maps.items():
            print(f"  Extracting {month} colors...")
            
            colors = []
            elevs = []
            valid_pixels = 0
            
            for x, y in sampled_pixels:
                scaled_x = int(x * x_scale)
                scaled_y = int(y * y_scale)
                
                if 0 <= scaled_x < monthly_width and 0 <= scaled_y < monthly_height:
                    color = monthly_map[scaled_y, scaled_x]
                    elev_x = min(int(x * x_scale), elevations.shape[1]-1)
                    elev_y = min(int(y * y_scale), elevations.shape[0]-1)
                    elevation = elevations[elev_y, elev_x]
                    
                    ocean_color = (2, 5, 20)
                    if color_distance(color, ocean_color) < 20:
                        continue
                    
                    colors.append(color)
                    elevs.append(elevation)
                    valid_pixels += 1
            
            if not colors:
                print(f"  Warning: No valid pixels found for {zone} in {month}")
                continue
                
            colors = np.array(colors)
            elevs = np.array(elevs)
            
            mean_color = np.mean(colors, axis=0).astype(int)
            median_color = np.median(colors, axis=0).astype(int)
            std_color = np.std(colors, axis=0).astype(int)
            p25_color = np.percentile(colors, 25, axis=0).astype(int)
            p75_color = np.percentile(colors, 75, axis=0).astype(int)
            
            if len(elevs) > 0:
                elevs_norm = elevs / 255.0
                low_mask = elevs_norm < 0.33
                mid_mask = (elevs_norm >= 0.33) & (elevs_norm < 0.66)
                high_mask = elevs_norm >= 0.66
                
                low_elev_color = np.mean(colors[low_mask], axis=0).astype(int) if np.any(low_mask) else None
                mid_elev_color = np.mean(colors[mid_mask], axis=0).astype(int) if np.any(mid_mask) else None
                high_elev_color = np.mean(colors[high_mask], axis=0).astype(int) if np.any(high_mask) else None
                
                elevation_colors = {
                    "low": low_elev_color.tolist() if low_elev_color is not None else None,
                    "mid": mid_elev_color.tolist() if mid_elev_color is not None else None,
                    "high": high_elev_color.tolist() if high_elev_color is not None else None
                }
            else:
                elevation_colors = {"low": None, "mid": None, "high": None}
            
            # Store seasonal data with hemisphere information
            zone_results[month] = {
                "mean": mean_color.tolist(),
                "median": median_color.tolist(),
                "std": std_color.tolist(),
                "p25": p25_color.tolist(),
                "p75": p75_color.tolist(),
                "elevation_colors": elevation_colors,
                "sample_count": valid_pixels,
                "tropical": tropical  # Mark if this is a tropical zone
            }
        
        results[zone] = zone_results
    
    return results

def generate_seasonal_palettes(colors_data):
    """
    Generate seasonal palettes by combining monthly data according to hemispheric seasons.
    Returns a dictionary with zone -> season -> color data.
    """
    seasonal_palettes = {}
    
    for zone, monthly_data in colors_data.items():
        zone_palettes = {}
        
        # Check if this is a tropical zone
        tropical = next(iter(monthly_data.values()))["tropical"]
        
        if tropical:
            # For tropical zones, we just average all months together
            all_colors = []
            all_elevations = []
            
            for month_data in monthly_data.values():
                all_colors.extend(month_data["mean"])
                all_elevations.extend(month_data["elevation_colors"])
            
            # Calculate averages
            mean_color = np.mean(all_colors, axis=0).astype(int).tolist()
            
            # Create a single "season" for tropical zones
            zone_palettes["tropical"] = {
                "mean": mean_color,
                "months": list(monthly_data.keys())  # All months included
            }
        else:
            # For non-tropical zones, process by hemisphere
            for hemisphere, season_months in SEASON_MAPPING.items():
                if hemisphere == 'tropical':
                    continue
                
                for season, months in season_months.items():
                    season_colors = []
                    season_elevations = []
                    
                    # Get data for each month in this season
                    for month in months:
                        if month in monthly_data:
                            month_data = monthly_data[month]
                            season_colors.append(month_data["mean"])
                    
                    if season_colors:
                        # Calculate seasonal average
                        mean_color = np.mean(season_colors, axis=0).astype(int).tolist()
                        
                        zone_palettes[f"{hemisphere}_{season}"] = {
                            "mean": mean_color,
                            "months": months
                        }
        
        seasonal_palettes[zone] = zone_palettes
    
    return seasonal_palettes

def generate_palette_visualization(colors_data, output_folder):
    """Generate visualization images showing seasonal palettes."""
    output_folder.mkdir(exist_ok=True, parents=True)
    all_zones = list(colors_data.keys())
    all_zones.sort()
    
    # Generate seasonal palettes
    seasonal_palettes = generate_seasonal_palettes(colors_data)
    
    # Create a visualization for each zone
    for zone in all_zones:
        zone_data = seasonal_palettes.get(zone, {})
        if not zone_data:
            continue
            
        # Create image (height based on number of seasons, width fixed)
        height = len(zone_data) * 50 + 50  # +50 for header
        width = 600
        vis_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add header
        vis_img[10:40, 10:300] = (220, 220, 220)
        
        # Draw zone name
        y_pos = 0
        
        # Draw each season
        for i, (season_name, season_data) in enumerate(zone_data.items()):
            y_pos = 50 + i * 50
            color = season_data["mean"]
            
            # Draw season name
            vis_img[y_pos+15:y_pos+35, 10:150] = (220, 220, 220)
            
            # Draw color block
            vis_img[y_pos+5:y_pos+45, 160:460] = color
            
            # Draw boundary lines
            vis_img[y_pos:y_pos+1, :] = (0, 0, 0)
        
        # Save zone visualization
        Image.fromarray(vis_img).save(output_folder / f"{zone}_palette.png")
        print(f"Saved visualization for {zone} to {output_folder / f'{zone}_palette.png'}")

def path_to_str(obj):
    """Convert Path objects to strings for JSON serialization"""
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def main():
    parser = argparse.ArgumentParser(
        description="Extract seasonal color palettes for Köppen-Geiger climate zones with hemispheric seasons",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--koppen-map", type=str,
                       default=str(FPATH_KOPPEN),
                       help="Path to Köppen-Geiger climate zone map (PNG format)")
    
    parser.add_argument("--elevation-map", type=str,
                       default=str(FPATH_ELEVATION),
                       help="Path to elevation map (grayscale JPG/PNG format)")
    
    parser.add_argument("--monthly-maps", nargs='+', type=str,
                       default=DEFAULT_MONTHLY_MAPS if DEFAULT_MONTHLY_MAPS else None,
                       help="Paths to monthly Earth maps (format: month:path, e.g., jan:earth_jan.jpg)")
    
    parser.add_argument("--output-json", type=str,
                       default=str(FPATH_OUTPUT),
                       help="Path to save the color palette JSON file")
    
    parser.add_argument("--output-viz", type=str, default=None,
                       help="Path to folder for saving visualization images (optional)")
    
    parser.add_argument("--tolerance", type=int, default=15,
                       help="Color tolerance for matching Köppen-Geiger zones")
    
    parser.add_argument("--sample-size", type=int, default=500,
                       help="Maximum number of pixels to sample per zone")
    
    args = parser.parse_args()
    
    # Load maps
    print(f"Loading Köppen-Geiger map from {args.koppen_map}")
    koppen_map = np.array(Image.open(args.koppen_map))
    koppen_height, koppen_width = koppen_map.shape[:2]
    print(f"Köppen map dimensions: {koppen_width}x{koppen_height}")
    
    # Calibrate colors
    calibrate_koppen_colors(koppen_map)
    
    # Load elevation map
    print(f"Loading elevation map from {args.elevation_map}")
    elevation_map = np.array(Image.open(args.elevation_map).convert('L'))
    elev_height, elev_width = elevation_map.shape[:2]
    print(f"Elevation map dimensions: {elev_width}x{elev_height}")
    
    # Load monthly maps
    monthly_maps = {}
    for month_path in args.monthly_maps:
        parts = month_path.split(':', 1)
        if len(parts) != 2:
            print(f"Warning: Invalid monthly map format: {month_path}. Use month:path format.")
            continue
            
        month, path = parts
        month = month.lower()
        
        print(f"Loading {month} map from {path}")
        monthly_map = np.array(Image.open(path))
        monthly_maps[month] = monthly_map
        
        height, width = monthly_map.shape[:2]
        print(f"{month.capitalize()} map dimensions: {width}x{height}")
    
    if not monthly_maps:
        print("Error: No valid monthly maps provided")
        return
    
    # Resize elevation map if needed
    first_monthly_map = next(iter(monthly_maps.values()))
    monthly_height, monthly_width = first_monthly_map.shape[:2]
    
    if elev_height != monthly_height or elev_width != monthly_width:
        print(f"Resizing elevation map to match monthly maps ({monthly_width}x{monthly_height})")
        elevation_pil = Image.fromarray(elevation_map)
        elevation_map = np.array(elevation_pil.resize((monthly_width, monthly_height), Image.LANCZOS))
    
    # Find climate zones
    zone_pixels = find_koppen_zones(koppen_map, tolerance=args.tolerance)
    debug_visualize_zones(koppen_map, zone_pixels, "zone_detection_debug.png")
    
    # Extract colors with hemispheric awareness
    colors_data = extract_seasonal_colors(
        koppen_map, 
        monthly_maps, 
        elevation_map, 
        zone_pixels,
        sample_size=args.sample_size
    )
    
    # Generate seasonal palettes
    seasonal_palettes = generate_seasonal_palettes(colors_data)
    
    # Add metadata
    metadata = {
        "koppen_map": args.koppen_map,
        "elevation_map": args.elevation_map,
        "monthly_maps": args.monthly_maps,
        "date_generated": np.datetime64('now').astype(str),
        "color_tolerance": args.tolerance,
        "sample_size": args.sample_size,
        "season_mapping": SEASON_MAPPING
    }
    
    # Create output structure
    output_data = {
        "metadata": metadata,
        "monthly_colors": colors_data,
        "seasonal_palettes": seasonal_palettes
    }
    
    # Save to JSON file
    output_path = Path(args.output_json)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=path_to_str)
    
    print(f"\nSaved color palette data to {output_path}")
    
    # Generate visualization if requested
    if args.output_viz:
        viz_folder = Path(args.output_viz)
        print(f"\nGenerating visualizations in {viz_folder}")
        generate_palette_visualization(colors_data, viz_folder)
    
    print("\nDone!")

if __name__ == "__main__":
    main()