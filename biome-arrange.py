import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# ===== Biome Definitions =====
# pgen biomes (RGB to biome name)
PGEN_COLORS = {
    (255, 255, 255): 'Ice',
    (210, 210, 210): 'Tundra',
    (250, 215, 165): 'Grasslands',
    (105, 155, 120): 'Taiga / Boreal Forest',
    (220, 195, 175): 'Desert',
    (225, 155, 100): 'Savanna',
    (155, 215, 170): 'Temperate Forest',
    (170, 195, 200): 'Temperate Rainforest',
    (185, 150, 160): 'Xeric Shrubland',
    (130, 190, 25): 'Tropical Dry Forest',
    (110, 160, 170): 'Tropical Rainforest'
}

# spacegeo biomes (RGB to biome name)
SPACEGEO_COLORS = {
    (151, 169, 173): 'Tundra',  # Ice/Tundra same color
    (99, 143, 82): 'Taiga / Boreal Forest',
    (29, 84, 109): 'Temperate Rainforest',
    (64, 138, 161): 'Temperate Seasonal Forest',
    (26, 82, 44): 'Tropical Rainforest',
    (174, 124, 11): 'Shrubland',
    (144, 126, 46): 'Temperate Grassland',
    (153, 165, 38): 'Savanna',
    (193, 113, 54): 'Subtropical Desert'
}

# = Köppen Mapping =====
BIOME_TO_KOPPEN = {
    # Tropical
    'Tropical Rainforest': 'Af',
    'Tropical Dry Forest': 'Aw',
    # Arid
    'Subtropical Desert': 'BWh',
    'Desert': 'BWk',  # Now defaults to cold desert
    'Xeric Shrubland': 'BSh',
    'Shrubland': 'BSk',  # More likely to be cold steppe
    'Temperate Grassland': 'BSk',
    'Savanna': 'Aw',
    # Temperate
    'Temperate Forest': 'Cfb',  # Default to oceanic
    'Temperate Seasonal Forest': 'Cfa',
    'Temperate Rainforest': 'Cfb',
    # Boreal/Continental
    'Taiga / Boreal Forest': 'Dfc',
    # Polar/Alpine
    'Tundra': 'ET',
    'Ice': 'EF'
}

KOPPEN_COLORS = {
    'Af': (0, 0, 254),     # Tropical rainforest - Dark blue
    'Am': (0, 119, 255),   # Tropical monsoon - Medium blue
    'Aw': (70, 169, 250),   # Tropical savanna - Light blue
    'As': (121, 186, 236),  # Tropical dry summer - Very light blue
    'BWh': (254, 0, 0),     # Hot desert - Red
    'BWk': (254, 150, 149), # Cold desert - Pink
    'BSh': (245, 163, 1),   # Hot semi-arid - Orange
    'BSk': (255, 219, 99),  # Cold semi-arid - Light yellow
    'Csa': (255, 255, 0),   # Hot-summer Mediterranean - Yellow
    'Csb': (198, 199, 0),   # Warm-summer Mediterranean - Olive
    'Csc': (150, 150, 0),   # Cold-summer Mediterranean - Dark olive
    'Cwa': (150, 255, 150), # Monsoon-influenced humid subtropical - Light green
    'Cwb': (99, 199, 100),  # Subtropical highland - Medium green
    'Cwc': (50, 150, 50),   # Cold subtropical highland - Dark green
    'Cfa': (198, 255, 78),  # Humid subtropical - Bright green
    'Cfb': (102, 255, 51),  # Oceanic - Lime green
    'Cfc': (51, 199, 1),    # Subpolar oceanic - Forest green
    'Dsa': (255, 0, 254),   # Hot-summer humid continental - Magenta
    'Dsb': (198, 0, 199),   # Warm-summer humid continental - Purple
    'Dsc': (150, 50, 149),  # Cold-summer humid continental - Dark purple
    'Dsd': (150, 100, 149), # Very cold winter humid continental - Very dark purple
    'Dwa': (171, 177, 255), # Monsoon-influenced hot-summer humid continental - Light blue-purple
    'Dwb': (90, 119, 219),  # Monsoon-influenced warm-summer humid continental - Medium blue-purple
    'Dwc': (76, 81, 181),   # Monsoon-influenced subarctic - Dark blue-purple
    'Dwd': (50, 0, 135),    # Monsoon-influenced extremely cold subarctic - Very dark blue
    'Dfa': (0, 255, 255),   # Hot-summer humid continental - Cyan
    'Dfb': (56, 199, 255),  # Warm-summer humid continental - Sky blue
    'Dfc': (0, 126, 125),   # Subarctic - Teal
    'Dfd': (0, 69, 94),     # Extremely cold subarctic - Dark teal
    'ET': (178, 178, 178),  # Tundra - Light gray
    'EF': (104, 104, 104),  # Ice cap - Dark gray
    'Ocean': (0, 0, 139)    # Deep ocean blue (added for water)
}


# ===== Core Functions =====
def closest_color(pixel, color_map):
    """Find the closest color in a palette."""
    r, g, b = pixel
    min_dist = float('inf')
    closest = None
    for color in color_map:
        cr, cg, cb = color
        # Convert to float before calculation to prevent overflow
        dist = float(r - cr)**2 + float(g - cg)**2 + float(b - cb)**2
        if dist < min_dist:
            min_dist = dist
            closest = color
    return closest

def get_biome(pixel, color_map):
    """Map RGB pixel to biome name."""
    closest = closest_color(pixel, color_map.keys())
    return color_map[closest]

def classify_koppen(pgen_pixel, spacegeo_pixel, elevation, lat_norm):
    """Classify a pixel into Köppen climate with improved latitude controls."""
    # Skip ocean pixels
    if tuple(spacegeo_pixel) == (76, 102, 178):
        return 'Ocean'
    
    # Get absolute latitude (0 at equator, 1 at poles)
    abs_lat = abs(lat_norm)
    
    # Get biomes from both maps
    pgen_biome = get_biome(pgen_pixel, PGEN_COLORS)
    spacegeo_biome = get_biome(spacegeo_pixel, SPACEGEO_COLORS)
    
    # Elevation override (alpine climates)
    if elevation > 0.8:  # Highest 20% of elevation
        if pgen_biome in ['Ice', 'Tundra']:
            return 'EF'
        return 'ET'
    
    # Merge biome priorities
    final_biome = pgen_biome
    if spacegeo_biome in ['Temperate Rainforest', 'Tropical Rainforest']:
        final_biome = spacegeo_biome
    
    # Köppen base class with latitude adjustments
    koppen = BIOME_TO_KOPPEN.get(final_biome, 'BSk')
    
    # ===== Enhanced Latitude Controls =====
    # Tropical Zone (0-23.5°)
    if abs_lat < 0.26:  # ~23.5° normalized
        if koppen in ['Af', 'Am', 'Aw']:
            pass  # Keep tropical classifications
        elif koppen == 'BWh':
            koppen = 'Aw' if elevation < 0.3 else 'Cwa'  # Transition to savanna/monsoon
        elif koppen == 'Cfa':
            koppen = 'Af' if elevation < 0.2 else 'Cwa'  # Force tropical if near equator
    
    # Subtropical Zone (23.5-35°)
    elif 0.26 <= abs_lat < 0.39:
        if koppen == 'Af':
            koppen = 'Am'  # Tropical monsoon more likely
        elif koppen == 'Cfb':
            koppen = 'Cfa'  # Humid subtropical more likely
    
    # Temperate Zone (35-55°)
    elif 0.39 <= abs_lat < 0.61:
        if koppen in ['Af', 'Am']:
            koppen = 'Cfa'  # Cannot have true tropics here
        elif koppen == 'Aw':
            koppen = 'Cwa'  # Transition to monsoon-influenced
        elif koppen == 'BWh':
            koppen = 'BSk'  # Desert becomes steppe
    
    # Boreal Zone (55-66.5°)
    elif 0.61 <= abs_lat < 0.74:
        if koppen in ['Cfa', 'Cfb']:
            koppen = 'Dfb'  # Transition to continental
        elif koppen == 'BSh':
            koppen = 'Dfc'  # Dry areas become subarctic
    
    # Arctic Zone (>66.5°)
    else:
        if koppen not in ['ET', 'EF', 'Dfc', 'Dfd']:
            koppen = 'ET'  # Force tundra/ice cap
    
    # Special case for rainforests outside tropics
    if final_biome == 'Tropical Rainforest' and abs_lat > 0.3:
        koppen = 'Cfb' if elevation < 0.4 else 'Dfb'
    
    return koppen

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
        heightmap = np.array(Image.open(heightmap_path).convert('L')).astype(float) / 255.0
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
                
            koppen_class = classify_koppen(
                pgen[y, x], 
                spacegeo[y, x], 
                heightmap[y, x],
                lat_norms[y]
            )
            if koppen_class is not None:  # Only assign if not ocean
                koppen_img[y, x] = KOPPEN_COLORS[koppen_class]
    
    print("\nSaving output...")
    with tqdm(desc="Saving Köppen map", unit="file") as pbar:
        Image.fromarray(koppen_img).save(output_path)
        pbar.update()
    
    print(f"\n✓ Köppen map saved to {output_path}")

# ===== CLI =====
if __name__ == "__main__":
    # Default file paths
    MAIN_WDIR = Path(r"D:\DND\Realistic DND World Gen\renders")
    DEFAULT_PGEN = MAIN_WDIR / "climate.bmp"
    DEFAULT_SPACEGEO = MAIN_WDIR / "canvas.png"
    DEFAULT_HEIGHTMAP = MAIN_WDIR / "greyscale linear.bmp"
    DEFAULT_OUTPUT = MAIN_WDIR / "koppen.bmp"

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
    
    args = parser.parse_args()
    
    generate_koppen_map(
        args.pgen, args.spacegeo, args.heightmap, 
        args.output, args.lat
    )