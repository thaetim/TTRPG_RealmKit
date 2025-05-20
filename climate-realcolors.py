import argparse
import numpy as np
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import os
import re
from pathlib import Path

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

# Define Koppen-Geiger color mappings (approximate RGB values from standard map)
# These are the expected colors in the Koppen-Geiger map
KOPPEN_COLORS = {
    'Af': (0, 0, 255),         # Tropical rainforest - Dark blue
    'Am': (0, 120, 255),       # Tropical monsoon - Medium blue
    'Aw': (70, 170, 250),      # Tropical savanna - Light blue
    'BWh': (255, 0, 0),        # Hot desert - Red
    'BWk': (255, 150, 150),    # Cold desert - Pink
    'BSh': (245, 165, 0),      # Hot semi-arid - Orange
    'BSk': (255, 220, 100),    # Cold semi-arid - Light yellow
    'Csa': (255, 255, 0),      # Hot-summer Mediterranean - Yellow
    'Csb': (200, 200, 0),      # Warm-summer Mediterranean - Olive
    'Csc': (150, 150, 0),      # Cold-summer Mediterranean - Dark olive
    'Cwa': (150, 255, 150),    # Monsoon-influenced humid subtropical - Light green
    'Cwb': (100, 200, 100),    # Subtropical highland - Medium green
    'Cwc': (50, 150, 50),      # Cold subtropical highland - Dark green
    'Cfa': (200, 255, 80),     # Humid subtropical - Bright green
    'Cfb': (100, 255, 50),     # Oceanic - Lime green
    'Cfc': (50, 200, 0),       # Subpolar oceanic - Forest green
    'Dsa': (255, 0, 255),      # Hot-summer humid continental - Magenta
    'Dsb': (200, 0, 200),      # Warm-summer humid continental - Purple
    'Dsc': (150, 50, 150),     # Cold-summer humid continental - Dark purple
    'Dsd': (150, 100, 150),    # Very cold winter humid continental - Very dark purple
    'Dwa': (170, 175, 255),    # Monsoon-influenced hot-summer humid continental - Light blue-purple
    'Dwb': (90, 120, 220),     # Monsoon-influenced warm-summer humid continental - Medium blue-purple
    'Dwc': (75, 80, 180),      # Monsoon-influenced subarctic - Dark blue-purple
    'Dwd': (50, 0, 135),       # Monsoon-influenced extremely cold subarctic - Very dark blue
    'Dfa': (0, 255, 255),      # Hot-summer humid continental - Cyan
    'Dfb': (55, 200, 255),     # Warm-summer humid continental - Sky blue
    'Dfc': (0, 125, 125),      # Subarctic - Teal
    'Dfd': (0, 70, 95),        # Extremely cold subarctic - Dark teal
    'ET': (180, 180, 180),     # Tundra - Light gray
    'EF': (105, 105, 105),     # Ice cap - Dark gray
}

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
    # Convert to numpy arrays if they aren't already
    c1 = np.array(c1)
    c2 = np.array(c2)
    
    # Calculate channel differences
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    
    # Weighted distance - more sensitive to hue than brightness
    return np.sqrt(
        2 * dr**2 +  # Red is important for climate zones
        4 * dg**2 +  # Green is very important for vegetation
        3 * db**2    # Blue is somewhat important
    )

def find_koppen_zones(koppen_map, tolerance=4):
    """Improved zone detection that handles interior pixels better"""
    print("Finding Koppen-Geiger zones in map...")
    height, width = koppen_map.shape[:2]
    
    # Create a reverse lookup from RGB to Koppen code
    rgb_to_koppen = {tuple(map(int, color)): code for code, color in KOPPEN_COLORS.items()}
    
    zone_pixels = defaultdict(list)
    black = (0, 0, 0)  # Boundary color
    
    # Pre-calculate all target colors as numpy arrays
    target_colors = [(np.array(color), code) for code, color in KOPPEN_COLORS.items()]

    # Define colors to EXCLUDE (oceans, boundaries, etc.)
    exclude_colors = {
        (0, 0, 0),        # Black boundaries
        (255, 255, 255),  # White oceans
        (102, 102, 102)   # Grey areas
    }
    
    with tqdm(total=height*width) as pbar:
        for y in range(height):
            for x in range(width):
                pixel = tuple(koppen_map[y, x][:3])
                
                # Skip excluded colors
                if pixel in exclude_colors:
                    pbar.update(1)
                    continue
                
                # Find closest matching zone
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
    
    # Filter out zones with too few pixels
    min_pixels = 50  # Increased minimum pixels
    filtered_zones = {zone: pixels for zone, pixels in zone_pixels.items() 
                     if len(pixels) >= min_pixels}
    
    print("\nZone detection results:")
    for zone, pixels in sorted(filtered_zones.items()):
        print(f"  {zone}: {len(pixels)} pixels")
    
    return filtered_zones

def debug_visualize_zones(koppen_map, zone_pixels, output_path="zone_debug.png"):
    """Improved debug visualization"""
    debug_img = koppen_map.copy()
    
    # Create zone color mapping
    zone_colors = KOPPEN_COLORS
    
    # Draw detected zones
    for zone, pixels in zone_pixels.items():
        color = zone_colors.get(zone, (255, 255, 0))  # Yellow for unknown
        for x, y in pixels:
            debug_img[y, x, :3] = color
    
    # Save with original and debug side-by-side
    comparison = Image.new('RGB', (koppen_map.shape[1]*2, koppen_map.shape[0]))
    comparison.paste(Image.fromarray(koppen_map), (0, 0))
    comparison.paste(Image.fromarray(debug_img), (koppen_map.shape[1], 0))
    comparison.save(output_path)
    print(f"Saved comparison visualization to {output_path}")

def debug_visualize_zones1(koppen_map, zone_pixels, output_path="debug_zones.png"):
    """Create a visualization of detected zones for debugging."""
    # Create a copy of the original image
    if koppen_map.shape[2] == 4:  # If RGBA
        debug_img = koppen_map.copy()
        # We'll modify only the RGB channels, keeping original alpha
    else:  # If RGB
        debug_img = koppen_map.copy()
    
    # Assign a unique color to each detected zone
    zone_colors = {}
    for i, zone in enumerate(zone_pixels.keys()):
        # Generate distinct colors (as RGB)
        hue = (i * 30) % 180 + 50  # Vary hue but keep it visible
        zone_colors[zone] = (hue, 200, 200)
    
    # Mark detected pixels
    for zone, pixels in zone_pixels.items():
        color = zone_colors[zone]
        for x, y in pixels:
            if debug_img.shape[2] == 4:  # RGBA
                debug_img[y, x, :3] = color  # Only modify RGB channels
            else:  # RGB
                debug_img[y, x] = color
    
    # Save the debug image
    Image.fromarray(debug_img).save(output_path)
    print(f"Debug visualization saved to {output_path}")

def resize_map(src_map, target_size):
    """
    Resize a map to match the target size, preserving aspect ratio.
    
    Parameters:
    -----------
    src_map : PIL.Image
        Source map image
    target_size : tuple
        Target size as (width, height)
        
    Returns:
    --------
    numpy.ndarray: Resized map as a numpy array
    """
    src_width, src_height = src_map.size
    target_width, target_height = target_size
    
    # Calculate scaling factors
    width_ratio = target_width / src_width
    height_ratio = target_height / src_height
    
    # Use the smaller ratio to ensure the entire map fits
    ratio = min(width_ratio, height_ratio)
    
    # Calculate new dimensions
    new_width = int(src_width * ratio)
    new_height = int(src_height * ratio)
    
    # Resize the image
    resized = src_map.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a black canvas of the target size
    result = Image.new("RGB", target_size, (0, 0, 0))
    
    # Calculate position to paste (center)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    
    # Paste the resized image
    result.paste(resized, (paste_x, paste_y))
    
    return np.array(result)

def extract_seasonal_colors(koppen_map, monthly_maps, elevations, zone_pixels, sample_size=500):
    """
    Extract representative colors for each Koppen zone from the monthly maps.
    
    Parameters:
    -----------
    koppen_map : numpy array
        RGB image array of the Koppen-Geiger map
    monthly_maps : dict
        Dictionary mapping month names to their image arrays
    elevations : numpy array
        Grayscale elevation data
    zone_pixels : dict
        Dictionary mapping Koppen zones to pixel coordinates
    sample_size : int
        Maximum number of pixels to sample per zone
        
    Returns:
    --------
    dict: Nested dictionary with zone -> month -> color statistics
    """
    koppen_height, koppen_width = koppen_map.shape[:2]
    
    # Get dimensions of monthly maps for scaling coordinates
    first_month = next(iter(monthly_maps.values()))
    monthly_height, monthly_width = first_month.shape[:2]
    
    # Calculate scaling factors
    x_scale = monthly_width / koppen_width
    y_scale = monthly_height / koppen_height
    
    results = {}
    
    # Process each zone
    for zone, pixels in zone_pixels.items():
        print(f"\nProcessing zone: {zone}")
        
        # Sample pixels if there are too many
        if len(pixels) > sample_size:
            sampled_pixels = np.random.choice(len(pixels), sample_size, replace=False)
            sampled_pixels = [pixels[i] for i in sampled_pixels]
        else:
            sampled_pixels = pixels
        
        zone_results = {}
        
        # Process each month
        for month, monthly_map in monthly_maps.items():
            print(f"  Extracting {month} colors...")
            
            # Collect colors from the monthly map at the zone pixel locations
            colors = []
            elevs = []
            valid_pixels = 0
            
            for x, y in sampled_pixels:
                # Scale coordinates to match monthly map dimensions
                scaled_x = int(x * x_scale)
                scaled_y = int(y * y_scale)
                
                # Ensure coordinates are within bounds
                if 0 <= scaled_x < monthly_width and 0 <= scaled_y < monthly_height:
                    # Get color and elevation
                    color = monthly_map[scaled_y, scaled_x]
                    
                    # Scale elevation coordinates similarly
                    elev_x = min(int(x * x_scale), elevations.shape[1]-1)
                    elev_y = min(int(y * y_scale), elevations.shape[0]-1)
                    elevation = elevations[elev_y, elev_x]
                    
                    # Skip if it's ocean (using approximate ocean color from instructions)
                    ocean_color = (2, 5, 20)  # hex #020514
                    if color_distance(color, ocean_color) < 20:
                        continue
                    
                    colors.append(color)
                    elevs.append(elevation)
                    valid_pixels += 1
            
            if not colors:
                print(f"  Warning: No valid pixels found for {zone} in {month}")
                continue
                
            # Convert to numpy for easier analysis
            colors = np.array(colors)
            elevs = np.array(elevs)
            
            # Calculate statistics
            mean_color = np.mean(colors, axis=0).astype(int)
            median_color = np.median(colors, axis=0).astype(int)
            std_color = np.std(colors, axis=0).astype(int)
            
            # Calculate percentiles for more robust color ranges
            p25_color = np.percentile(colors, 25, axis=0).astype(int)
            p75_color = np.percentile(colors, 75, axis=0).astype(int)
            
            # Calculate colors by elevation bands
            if len(elevs) > 0:
                # Normalize elevations to 0-1
                elevs_norm = elevs / 255.0
                
                # Create elevation bands
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
            
            # Store results
            zone_results[month] = {
                "mean": mean_color.tolist(),
                "median": median_color.tolist(),
                "std": std_color.tolist(),
                "p25": p25_color.tolist(),
                "p75": p75_color.tolist(),
                "elevation_colors": elevation_colors,
                "sample_count": valid_pixels
            }
        
        # Store results for this zone
        results[zone] = zone_results
    
    return results

def generate_palette_visualization(colors_data, output_folder):
    """
    Generate visualization images for each month showing the extracted colors.
    
    Parameters:
    -----------
    colors_data : dict
        Nested dictionary with zone -> month -> color statistics
    output_folder : Path
        Folder to save visualization images
    """
    # Create output folder if it doesn't exist
    output_folder.mkdir(exist_ok=True, parents=True)
    
    # Extract all zones and months
    all_zones = list(colors_data.keys())
    all_zones.sort()  # Sort alphabetically 
    
    all_months = set()
    for zone_data in colors_data.values():
        all_months.update(zone_data.keys())
    all_months = sorted(list(all_months))
    
    # Generate a visualization for each month
    for month in all_months:
        # Create image (height based on number of zones, width fixed)
        height = len(all_zones) * 50
        width = 600
        vis_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        for i, zone in enumerate(all_zones):
            # Skip if zone doesn't have data for this month
            if month not in colors_data[zone]:
                continue
                
            zone_data = colors_data[zone][month]
            
            # Get colors
            mean_color = tuple(zone_data["mean"])
            p25_color = tuple(zone_data["p25"])
            p75_color = tuple(zone_data["p75"])
            
            # Define y-positions
            y_pos = i * 50
            
            # Draw zone code
            vis_img[y_pos+15:y_pos+35, 10:150] = (220, 220, 220)
            
            # Draw color blocks
            # Mean color
            vis_img[y_pos+5:y_pos+45, 160:260] = mean_color
            
            # 25th percentile color
            vis_img[y_pos+5:y_pos+45, 270:370] = p25_color
            
            # 75th percentile color 
            vis_img[y_pos+5:y_pos+45, 380:480] = p75_color
            
            # Draw boundary lines
            vis_img[y_pos:y_pos+1, :] = (0, 0, 0)
        
        # Add legend at top
        legend_img = np.ones((50, width, 3), dtype=np.uint8) * 240
        legend_text_positions = [(10, 30), (160, 30), (270, 30), (380, 30)]
        legend_texts = ["Zone", "Mean", "25th %ile", "75th %ile"]
        
        # Draw legend texts background
        for (x, y), text in zip(legend_text_positions, legend_texts):
            legend_img[y-20:y+5, x:x+100] = (220, 220, 220)
        
        # Add legend to top of visualization
        result_img = np.vstack([legend_img, vis_img])
        
        # Save as image
        Image.fromarray(result_img).save(output_folder / f"{month}_palette.png")
        print(f"Saved visualization for {month} to {output_folder / f'{month}_palette.png'}")

# Path serializer function
def path_to_str(obj):
    """Convert Path objects to strings for JSON serialization"""
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def main():
    parser = argparse.ArgumentParser(
        description="Extract seasonal color palettes for Köppen-Geiger climate zones",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--koppen-map", type=str,
                       default=str(FPATH_KOPPEN),  # Convert Path to string
                       help="Path to Köppen-Geiger climate zone map (PNG format)")
    
    parser.add_argument("--elevation-map", type=str,
                       default=str(FPATH_ELEVATION),  # Convert Path to string
                       help="Path to elevation map (grayscale JPG/PNG format)")
    
    parser.add_argument("--monthly-maps", nargs='+', type=str,
                       default=DEFAULT_MONTHLY_MAPS if DEFAULT_MONTHLY_MAPS else None,
                       help="Paths to monthly Earth maps (format: month:path, e.g., jan:earth_jan.jpg)")
    
    parser.add_argument("--output-json", type=str,
                       default=str(FPATH_OUTPUT),  # Convert Path to string
                       help="Path to save the color palette JSON file")
    
    parser.add_argument("--output-viz", type=str, default=None,
                       help="Path to folder for saving visualization images (optional)")
    
    parser.add_argument("--tolerance", type=int, default=15,
                       help="Color tolerance for matching Köppen-Geiger zones")
    
    parser.add_argument("--sample-size", type=int, default=500,
                       help="Maximum number of pixels to sample per zone")
    
    args = parser.parse_args()
    
    # Load Köppen-Geiger map
    print(f"Loading Köppen-Geiger map from {args.koppen_map}")
    koppen_map = np.array(Image.open(args.koppen_map))
    koppen_height, koppen_width = koppen_map.shape[:2]
    print(f"Köppen map dimensions: {koppen_width}x{koppen_height}")
    # Calibrate
    calibrate_koppen_colors(koppen_map)
    print("Checking calibrated sample pixel values...")
    sample_pixels = [
        koppen_map[100, 100],  # Sample from top-left quadrant
        koppen_map[100, koppen_width//2],  # Sample from top-center
        koppen_map[koppen_height//2, koppen_width//2],  # Sample from center
        koppen_map[-100, -100]  # Sample from bottom-right quadrant
    ]
    print("Sample pixel values:", sample_pixels)
    print("\nActual colors in your map (sampling 10,000 points):")
    sample_colors = defaultdict(int)
    for _ in range(10000):
        y, x = np.random.randint(0, koppen_map.shape[0]), np.random.randint(0, koppen_map.shape[1])
        color = tuple(koppen_map[y,x][:3])
        if color != (0,0,0):  # Ignore black boundaries
            sample_colors[color] += 1
    print("Top 10 most common interior colors:")
    for color, count in sorted(sample_colors.items(), key=lambda x: -x[1])[:10]:
        print(f"RGB{color}: {count} pixels (hex: #{color[0]:02x}{color[1]:02x}{color[2]:02x})")
    
    # Load elevation map
    print(f"Loading elevation map from {args.elevation_map}")
    elevation_map = np.array(Image.open(args.elevation_map).convert('L'))
    
    # Ensure elevation map has same dimensions as monthly maps
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
        
        # Store in dictionary
        monthly_maps[month] = monthly_map
        
        # Report dimensions
        height, width = monthly_map.shape[:2]
        print(f"{month.capitalize()} map dimensions: {width}x{height}")
    
    # Make sure we have at least one monthly map
    if not monthly_maps:
        print("Error: No valid monthly maps provided")
        return
    
    # Ensure all maps are consistent in dimension
    first_monthly_map = next(iter(monthly_maps.values()))
    monthly_height, monthly_width = first_monthly_map.shape[:2]
    
    # Check if elevation map needs resizing
    if elev_height != monthly_height or elev_width != monthly_width:
        print(f"Resizing elevation map to match monthly maps ({monthly_width}x{monthly_height})")
        elevation_pil = Image.fromarray(elevation_map)
        elevation_map = np.array(elevation_pil.resize((monthly_width, monthly_height), Image.LANCZOS))
    
    # Find Koppen-Geiger zones
    zone_pixels = find_koppen_zones(koppen_map, tolerance=args.tolerance)
    debug_visualize_zones(koppen_map, zone_pixels, "zone_detection_debug.png")
    
    # Extract colors from monthly maps
    colors_data = extract_seasonal_colors(
        koppen_map, 
        monthly_maps, 
        elevation_map, 
        zone_pixels,
        sample_size=args.sample_size
    )
    
    # Add metadata
    metadata = {
        "koppen_map": args.koppen_map,
        "elevation_map": args.elevation_map,
        "monthly_maps": args.monthly_maps,
        "date_generated": np.datetime64('now').astype(str),
        "color_tolerance": args.tolerance,
        "sample_size": args.sample_size
    }
    
    # Create output structure
    output_data = {
        "metadata": metadata,
        "colors": colors_data
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