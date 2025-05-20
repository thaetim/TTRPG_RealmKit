import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

from biome_arrange_constants import *

def load_color_data(json_path):
    """Load color data from JSON file and reformat it for our use"""
    with open(json_path) as f:
        data = json.load(f)
    
    # Reformat the data to match our expected structure
    color_data = {}
    for koppen_code, monthly_data in data['colors'].items():
        color_data[koppen_code] = {
            'jan': monthly_data['jan']['mean'],
            'apr': monthly_data['apr']['mean'],
            'jul': monthly_data['jul']['mean'],
            'oct': monthly_data['oct']['mean']
        }
    
    return color_data

def enhance_color_realism(colors, season):
    """Apply color theory and natural variation to make colors more realistic"""
    enhanced_colors = {}
    
    for koppen_code, color_values in colors.items():
        if season in color_values:
            base_color = color_values[season]
            
            # Add slight variations to create more natural look
            variation = np.random.randint(-5, 6, size=3)
            varied_color = np.clip(np.array(base_color) + variation, 0, 255)
            
            # Store enhanced color
            enhanced_colors[koppen_code] = tuple(varied_color)
            
    return enhanced_colors

def apply_realistic_colors(koppen_map_path, heightmap_path, output_path, season="summer", 
                          color_data=None, add_variation=True, json_path=None):
    """
    Apply realistic Earth-like colors to a Köppen climate map based on the season.
    
    Parameters:
    -----------
    koppen_map_path : str or Path
        Path to the Köppen climate map image
    heightmap_path : str or Path
        Path to the heightmap image (used for terrain shading)
    output_path : str or Path
        Path to save the output image
    season : str
        One of "jan", "apr", "jul", "oct" (representing seasons)
    color_data : dict, optional
        Custom color data to use instead of default
    add_variation : bool
        Whether to add color variation for more realism
    json_path : str or Path, optional
        Path to JSON file containing color data
    """
    print(f"\nGenerating realistic {season} biome map...")
    
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
    
    # Create shading factors from heightmap
    # Higher elevation = slightly lighter color
    elevation_factor = 0.7 + (heightmap * 0.5)
    
    # Process pixels
    with tqdm(total=height*width, desc="Applying realistic colors") as pbar:
        for y in range(height):
            for x in range(width):
                # Get Köppen class
                pixel_tuple = tuple(koppen_img[y, x])
                koppen_class = COLOR_TO_KOPPEN.get(pixel_tuple, 'Ocean')
                
                # Get base color for this biome + season
                if koppen_class == 'Ocean':
                    # Ocean blues vary by depth
                    depth_factor = max(0.3, 1.0 - heightmap[y, x] * 2)
                    base_color = (int(0 * depth_factor), 
                                 int(80 * depth_factor), 
                                 int(160 * depth_factor))
                else:
                    # Land biomes
                    if koppen_class in colors and season in colors[koppen_class]:
                        base_color = colors[koppen_class][season]
                    else:
                        # Fallback to a default color if not defined
                        base_color = (150, 150, 100)  # Generic land color
                
                # Apply elevation shading
                elev_factor = elevation_factor[y, x]
                shaded_color = tuple(min(255, int(c * elev_factor)) for c in base_color)
                
                # Add slight noise for natural variation (only on land)
                if koppen_class != 'Ocean' and add_variation:
                    color_noise = np.random.randint(-5, 6, size=3)
                    final_color = np.clip(np.array(shaded_color) + color_noise, 0, 255)
                    realistic_img[y, x] = final_color
                else:
                    realistic_img[y, x] = shaded_color
                
                pbar.update(1)
    
    # Save output
    Image.fromarray(realistic_img).save(output_path)
    print(f"\n✓ Realistic {season} biome map saved to {output_path}")

def generate_all_seasons(koppen_map_path, heightmap_path, output_dir, add_variation=True, json_path=None):
    """Generate realistic biome maps for all four seasons"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load color data from JSON if provided
    if json_path:
        print("Loading realistic Earth biome colors from JSON...")
        color_data = load_color_data(json_path)
    else:
        print("No color data provided, using fallback colors")
        color_data = {}
    
    # Generate each season
    for season in ["jan", "apr", "jul", "oct"]:
        output_path = output_dir / f"realistic_biomes_{season}.png"
        apply_realistic_colors(
            koppen_map_path, 
            heightmap_path, 
            output_path,
            season=season,
            color_data=color_data,
            add_variation=add_variation,
            json_path=None  # Already loaded the data
        )

if __name__ == "__main__":
    # Default file paths
    MAIN_WDIR = Path(r"D:\DND\Realistic DND World Gen")
    DEFAULT_HEIGHTMAP = MAIN_WDIR / "renders" / "greyscale tuned.bmp"
    DEFAULT_KOPPEN = MAIN_WDIR / "renders" / "koppen tuned.bmp"
    DEFAULT_OUTPUT = MAIN_WDIR / "renders" / "colorized tuned.bmp"
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
            json_path=args.json
        )
    else:
        output_path = output_dir / f"realistic_biomes_{args.season}.png"
        apply_realistic_colors(
            args.koppen, 
            args.heightmap, 
            output_path,
            season=args.season,
            add_variation=not args.no_variation,
            json_path=args.json
        )