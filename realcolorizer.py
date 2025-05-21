import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

from biome_arrange_constants import *

def rgb_to_hsv(rgb):
    """Convert RGB color (0-1 range) to HSV (returns numpy array)"""
    rgb = np.array(rgb)
    maxc = rgb.max()
    minc = rgb.min()
    v = maxc
    if minc == maxc:
        return np.array([0.0, 0.0, v])
    s = (maxc-minc) / maxc
    rc = (maxc-rgb[0]) / (maxc-minc)
    gc = (maxc-rgb[1]) / (maxc-minc)
    bc = (maxc-rgb[2]) / (maxc-minc)
    if rgb[0] == maxc:
        h = bc-gc
    elif rgb[1] == maxc:
        h = 2.0+rc-bc
    else:
        h = 4.0+gc-rc
    h = (h/6.0) % 1.0
    return np.array([h, s, v])

def hsv_to_rgb(hsv):
    """Convert HSV color (0-1 range) to RGB (accepts numpy array)"""
    h, s, v = hsv
    if s == 0.0:
        return np.array([v, v, v])
    i = int(h*6.0)
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    if i == 0:
        return np.array([v, t, p])
    if i == 1:
        return np.array([q, v, p])
    if i == 2:
        return np.array([p, v, t])
    if i == 3:
        return np.array([p, q, v])
    if i == 4:
        return np.array([t, p, v])
    if i == 5:
        return np.array([v, p, q])

def load_color_data(json_path):
    """Load color data from JSON file with enhanced color selection"""
    with open(json_path) as f:
        data = json.load(f)
    
    color_data = {}
    for koppen_code, monthly_data in data['colors'].items():
        color_data[koppen_code] = {}
        for month, month_data in monthly_data.items():
            # Convert to numpy arrays
            mean_color = np.array(month_data['mean'])
            p75_color = np.array(month_data['p75'])
            
            # Blend towards the more vibrant color
            base_color = mean_color * 0.3 + p75_color * 0.7
            
            # Convert to HSV and adjust saturation
            hsv = rgb_to_hsv(base_color/255.0)
            hsv[1] = min(1.0, hsv[1]*1.2)  # 20% saturation boost
            enhanced_color = hsv_to_rgb(hsv)*255
            
            # Convert back to list for JSON compatibility
            color_data[koppen_code][month] = enhanced_color.astype(int).tolist()
            
    return color_data

def get_season_by_hemisphere(y, height, season):
    """Adjust season based on hemisphere (y-coordinate)"""
    if y < height // 2:  # Northern hemisphere
        return season
    else:  # Southern hemisphere - invert seasons
        season_map = {
            "jan": "jul",
            "apr": "oct",
            "jul": "jan",
            "oct": "apr"
        }
        return season_map.get(season, season)

def enhance_color_realism(colors, season):
    """Apply color theory and natural variation to make colors more realistic"""
    enhanced_colors = {}
    
    for koppen_code, seasonal_colors in colors.items():
        enhanced_colors[koppen_code] = {}
        
        # Use annual average if specific season is missing
        if season in seasonal_colors:
            base_color = seasonal_colors[season]
        else:
            # Fallback to average of all available seasons
            all_season_colors = list(seasonal_colors.values())
            base_color = np.mean(all_season_colors, axis=0) if all_season_colors else np.array([120, 100, 60])
            
        # Add variation even if using fallback
        variation = np.random.randint(-5, 6, size=3)
        varied_color = np.clip(np.array(base_color) + variation, 0, 255)
        
        enhanced_colors[koppen_code][season] = tuple(varied_color)
            
    return enhanced_colors

def apply_realistic_colors(koppen_map_path, heightmap_path, output_path, season="summer", 
                         color_data=None, add_variation=True, json_path=None, skip_ocean=False):
    print(f"\nGenerating realistic {season} biome map...")
    if skip_ocean:
        print("Skipping ocean color processing")
    
    # Load color data
    if json_path:
        colors = load_color_data(json_path)
    else:
        colors = color_data if color_data else {}
    
    # Load maps
    koppen_img = np.array(Image.open(koppen_map_path))
    heightmap = np.array(Image.open(heightmap_path).convert('L')) / 255.0
    
    height, width = heightmap.shape
    realistic_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Enhance colors for realism if requested
    if add_variation:
        colors = enhance_color_realism(colors, season)
    else:
        # Maintain full structure when not enhancing
        colors = {k: {season: v[season]} for k, v in colors.items() if season in v}
    
    # Process pixels
    missing_classes = set()
    missing_seasons = set()
    with tqdm(total=height*width, desc="Applying realistic colors") as pbar:
        for y in range(height):
            current_season = get_season_by_hemisphere(y, height, season)
            for x in range(width):
                # Get Köppen class
                pixel_tuple = tuple(koppen_img[y, x])
                koppen_class = COLOR_TO_KOPPEN.get(pixel_tuple, 'Ocean')
                
                # Skip ocean processing if requested
                if skip_ocean and koppen_class == 'Ocean':
                    realistic_img[y, x] = koppen_img[y, x]  # Keep original ocean color
                    pbar.update(1)
                    continue
                
                # Get base color for this biome + season
                if koppen_class == 'Ocean':
                    # Ocean blues vary by depth
                    depth_factor = max(0.3, 1.0 + heightmap[y, x] * 1.5)
                    base_color = np.array([
                        int(20 * depth_factor), 
                        int(100 * depth_factor), 
                        int(200 * depth_factor)
                    ])
                else:
                    # Land biomes
                    try:
                        # Use the season adjusted for hemisphere
                        base_color = np.array(colors[koppen_class][current_season])
                    except (KeyError, TypeError) as e:
                        # Track missing data instead of raising errors
                        if koppen_class not in colors:
                            missing_classes.add(koppen_class)
                        elif season not in colors[koppen_class]:
                            missing_seasons.add(f"{koppen_class}:{season}")
                        else:
                            # For other unexpected errors, use fallback color but log it
                            missing_classes.add(f"ERROR:{koppen_class}:{str(e)}")
                            base_color = np.array([120, 100, 60])  # Fallback color
                        
                        # Use fallback color if we got here
                        if 'base_color' not in locals():
                            base_color = np.array([120, 100, 60])  # Fallback color
                
                # Apply elevation shading (now inside the loop)
                elev_factor = 0.85 + (heightmap[y, x] * 0.3)
                shaded_color = base_color * elev_factor
                
                # Add slight noise for natural variation
                if koppen_class != 'Ocean' and add_variation:
                    color_noise = np.random.randint(-3, 4, size=3)
                    final_color = np.clip(shaded_color + color_noise, 0, 255)
                else:
                    final_color = shaded_color
                
                # Apply saturation boost to land areas
                if koppen_class != 'Ocean':
                    hsv = rgb_to_hsv(final_color/255.0)
                    hsv[1] = min(1.0, hsv[1]*1.3)
                    final_color = hsv_to_rgb(hsv)*255
                
                realistic_img[y, x] = final_color.astype(np.uint8)
                pbar.update(1)

    # After processing all pixels, save missing data to file
    if missing_classes or missing_seasons:
        with open("missing_color_data.txt", "w") as f:
            if missing_classes:
                f.write("Missing Köppen classes:\n")
                f.write("\n".join(sorted(missing_classes)) + "\n\n")
            if missing_seasons:
                f.write("Missing season data (format: KöppenClass:Season):\n")
                f.write("\n".join(sorted(missing_seasons)) + "\n")
    
    # Save output
    Image.fromarray(realistic_img).save(output_path)
    print(f"\n✓ Realistic {season} biome map saved to {output_path}")

def generate_all_seasons(koppen_map_path, heightmap_path, output_dir, add_variation=True, json_path=None, skip_ocean=False):
    """Generate realistic biome maps for all four seasons with hemisphere awareness"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load color data
    if json_path:
        print("Loading realistic Earth biome colors from JSON...")
        colors = load_color_data(json_path)
    else:
        print("No color data provided, using fallback colors")
        colors = {}
    
    # Load maps
    koppen_img = np.array(Image.open(koppen_map_path))
    heightmap = np.array(Image.open(heightmap_path).convert('L')) / 255.0
    height, width = heightmap.shape
    
    # Generate each season
    for season in ["jan", "apr", "jul", "oct"]:
        output_path = output_dir / f"realistic_biomes_{season}.png"
        realistic_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        missing_classes = set()
        missing_seasons = set()
        
        with tqdm(total=height*width, desc=f"Generating {season} map") as pbar:
            for y in range(height):
                # Determine season based on hemisphere
                current_season = get_season_by_hemisphere(y, height, season)
                
                for x in range(width):
                    # Get Köppen class
                    pixel_tuple = tuple(koppen_img[y, x])
                    koppen_class = COLOR_TO_KOPPEN.get(pixel_tuple, 'Ocean')
                    
                    # Skip ocean processing if requested
                    if skip_ocean and koppen_class == 'Ocean':
                        realistic_img[y, x] = koppen_img[y, x]  # Keep original ocean color
                        pbar.update(1)
                        continue
                    
                    # Get base color for this biome + season
                    if koppen_class == 'Ocean':
                        # Ocean blues vary by depth
                        depth_factor = max(0.3, 1.0 + heightmap[y, x] * 1.5)
                        base_color = np.array([
                            int(20 * depth_factor), 
                            int(100 * depth_factor), 
                            int(200 * depth_factor)
                        ])
                    else:
                        # Land biomes
                        try:
                            base_color = np.array(colors[koppen_class][current_season])
                        except (KeyError, TypeError) as e:
                            # Track missing data
                            if koppen_class not in colors:
                                missing_classes.add(koppen_class)
                            elif current_season not in colors[koppen_class]:
                                missing_seasons.add(f"{koppen_class}:{current_season}")
                            else:
                                missing_classes.add(f"ERROR:{koppen_class}:{str(e)}")
                            
                            base_color = np.array([120, 100, 60])  # Fallback color
                    
                    # Apply elevation shading
                    elev_factor = 0.85 + (heightmap[y, x] * 0.3)
                    shaded_color = base_color * elev_factor
                    
                    # Add slight noise for natural variation
                    if koppen_class != 'Ocean' and add_variation:
                        color_noise = np.random.randint(-3, 4, size=3)
                        final_color = np.clip(shaded_color + color_noise, 0, 255)
                    else:
                        final_color = shaded_color
                    
                    # Apply saturation boost to land areas
                    if koppen_class != 'Ocean':
                        hsv = rgb_to_hsv(final_color/255.0)
                        hsv[1] = min(1.0, hsv[1]*1.3)
                        final_color = hsv_to_rgb(hsv)*255
                    
                    realistic_img[y, x] = final_color.astype(np.uint8)
                    pbar.update(1)

        # Save missing data to file
        if missing_classes or missing_seasons:
            missing_file = output_dir / f"missing_color_data_{season}.txt"
            with open(missing_file, "w") as f:
                if missing_classes:
                    f.write("Missing Köppen classes:\n")
                    f.write("\n".join(sorted(missing_classes)) + "\n\n")
                if missing_seasons:
                    f.write("Missing season data (format: KöppenClass:Season):\n")
                    f.write("\n".join(sorted(missing_seasons)) + "\n")
        
        # Save output
        Image.fromarray(realistic_img).save(output_path)
        print(f"\n✓ Realistic {season} biome map saved to {output_path}")

if __name__ == "__main__":
    # Default file paths
    MAIN_WDIR = Path(r"D:\DND\Realistic DND World Gen")
    DEFAULT_HEIGHTMAP = MAIN_WDIR / "renders" / "greyscale tuned.bmp"
    DEFAULT_KOPPEN = MAIN_WDIR / "renders" / "koppen tuned.bmp"
    DEFAULT_OUTPUT = MAIN_WDIR / "renders"
    DEFAULT_CLIMATE_PALETTES = MAIN_WDIR / "earth" / "climate_palettes.json"

    parser = argparse.ArgumentParser(
        description="Generate realistic Earth-like biome maps based on Köppen climate classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--koppen", default=DEFAULT_KOPPEN,
                       help="Input Köppen climate map (from koppen_generator.py)")
    parser.add_argument("--heightmap", default=DEFAULT_HEIGHTMAP,
                       help="Heightmap (grayscale)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                       help="Output directory for realistic biome maps")
    parser.add_argument("--season", choices=["jan", "apr", "jul", "oct"], default="jul",
                       help="Which season to generate (if not generating all)")
    parser.add_argument("--all-seasons", action="store_true",
                       help="Generate maps for all four seasons")
    parser.add_argument("--no-variation", action="store_true",
                       help="Disable random color variation for more consistent output")
    parser.add_argument("--skip-ocean", action="store_true",
                       help="Skip ocean processing and keep original ocean colors")
    parser.add_argument("--json", default=DEFAULT_CLIMATE_PALETTES,
                       help="Path to JSON file containing color data")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.all_seasons:
        generate_all_seasons(
            args.koppen, 
            args.heightmap, 
            output_dir,
            add_variation=not args.no_variation,
            json_path=args.json,
            skip_ocean=args.skip_ocean
        )
    else:
        output_path = output_dir / f"realistic_biomes_{args.season}.png"
        apply_realistic_colors(
            args.koppen, 
            args.heightmap, 
            output_path,
            season=args.season,
            add_variation=not args.no_variation,
            json_path=args.json,
            skip_ocean=args.skip_ocean
        )