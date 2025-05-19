import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from functools import lru_cache
import numpy as np

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

def classify_koppen(pgen_pixel, spacegeo_pixel, elevation, lat_norm):
    """Final refined Köppen classification based on comprehensive analysis."""
    # Convert normalized latitude to degrees
    abs_lat = abs(lat_norm * 90)  # 0° to 90°
    
    # ===== Improved Ocean Detection =====
    ocean_colors = [
        (76, 102, 178),  # SpaceGeo ocean
        (0, 0, 139),      # Köppen ocean
        (110, 160, 170)   # PGen ocean-like
    ]
    if (tuple(spacegeo_pixel) in ocean_colors or 
        tuple(pgen_pixel) in ocean_colors or
        elevation < 0.005):  # Absolute water threshold
        return 'Ocean'
    
    # ===== Elevation Normalization =====
    # Adjust elevation to account for minimum observed 0.502
    adj_elev = (elevation - 0.5) * 2 if elevation > 0.5 else 0
    adj_elev = max(0, min(1, adj_elev))  # Clamp to 0-1
    
    # ===== Get Biomes =====
    pgen_biome = get_biome(pgen_pixel, PGEN_COLORS)
    spacegeo_biome = get_biome(spacegeo_pixel, SPACEGEO_COLORS)
    
    # ===== Alpine Climate Override =====
    if adj_elev > 0.9:
        return 'EF' if (adj_elev > 0.95 or pgen_biome == 'Ice') else 'ET'
    
    # ===== Fix Specific Issues =====
    # 1. Cold Steppe (BSk) overrepresentation
    if (pgen_biome == 'Grasslands' and 
        spacegeo_biome in ['Tundra', 'Temperate Grassland'] and
        abs_lat > 30):
        return 'BSk'
    
    # 2. Temperate Forest misclassifications
    if (pgen_biome == 'Temperate Forest' and 
        spacegeo_biome == 'Tundra' and
        abs_lat < 60):
        return 'Dfc' if adj_elev > 0.4 else 'Cfb'
    
    # ===== Climate Zones =====
    # Tropical (0-23.5°)
    if abs_lat < 23.5:
        if spacegeo_biome == 'Tropical Rainforest':
            return 'Af'
        elif pgen_biome == 'Tropical Dry Forest':
            return 'Aw'
        elif spacegeo_biome == 'Subtropical Desert':
            return 'BWh'
        elif pgen_biome == 'Desert':
            return 'BWh' if adj_elev < 0.6 else 'BWk'
    
    # Subtropical (23.5-35°)
    elif 23.5 <= abs_lat < 35:
        if pgen_biome == 'Xeric Shrubland':
            return 'BSh'
        elif spacegeo_biome == 'Temperate Seasonal Forest':
            return 'Cfa'
        elif pgen_biome == 'Grasslands':
            return 'BSk' if adj_elev > 0.3 else 'Cwa'
    
    # Temperate (35-55°)
    elif 35 <= abs_lat < 55:
        if spacegeo_biome == 'Temperate Rainforest':
            return 'Cfb'
        elif pgen_biome == 'Temperate Forest':
            return 'Dfb' if adj_elev > 0.5 else 'Cfb'
        elif pgen_biome == 'Taiga / Boreal Forest':
            return 'Dfc'
    
    # Boreal (55-66.5°)
    elif 55 <= abs_lat < 66.5:
        if pgen_biome in ['Taiga / Boreal Forest', 'Temperate Forest']:
            return 'Dfc'
        elif pgen_biome == 'Tundra':
            return 'ET'
    
    # Arctic (>66.5°)
    else:
        return 'EF' if pgen_biome == 'Ice' else 'ET'
    
    # ===== Fallback Rules =====
    # Priority to spacegeo for specific biomes
    if spacegeo_biome in SPACEGEO_COLORS:
        if spacegeo_biome in BIOME_TO_KOPPEN:
            return BIOME_TO_KOPPEN[spacegeo_biome]
    
    # Default to pgen biome mapping
    return BIOME_TO_KOPPEN.get(pgen_biome, 'BSk')  # Default to cold steppe

def classify_koppen5(pgen_pixel, spacegeo_pixel, elevation, lat_norm):
    """Refined Köppen classification based on distribution analysis."""
    # Skip ocean pixels - more robust detection
    if (tuple(spacegeo_pixel) == (76, 102, 178) or 
        tuple(pgen_pixel) == (76, 102, 178) or
        elevation < 0.001):  # Absolute water level
        return 'Ocean'
    
    abs_lat = abs(lat_norm * 90)  # Convert to degrees
    elevation = max(0, min(1, elevation))  # Clamp elevation
    
    # Get biomes from both maps
    pgen_biome = get_biome(pgen_pixel, PGEN_COLORS)
    spacegeo_biome = get_biome(spacegeo_pixel, SPACEGEO_COLORS)
    
    # ===== Elevation Overrides =====
    if elevation > 0.8:
        if elevation > 0.95 or pgen_biome == 'Ice':
            return 'EF'
        return 'ET'
    
    # ===== Fix Problematic Combinations =====
    # 1. Tropical Dry Forest + Tundra (shouldn't exist)
    if (pgen_biome == 'Tropical Dry Forest' and 
        spacegeo_biome == 'Tundra'):
        return 'Aw' if abs_lat < 15 else 'Cwa'
    
    # 2. Temperate Forest + Tundra (arctic transition)
    if (pgen_biome == 'Temperate Forest' and 
        spacegeo_biome == 'Tundra'):
        return 'ET' if abs_lat > 60 else 'Dfc'
    
    # ===== Climate Zone Determination =====
    # Tropical Zone (0-23.5°)
    if abs_lat < 23.5:
        if spacegeo_biome == 'Tropical Rainforest':
            return 'Af'
        elif pgen_biome == 'Tropical Dry Forest':
            return 'Am' if elevation < 0.3 else 'Aw'
        elif pgen_biome == 'Savanna':
            return 'Aw'
        elif pgen_biome == 'Desert':
            return 'BWh' if elevation < 0.6 else 'BWk'
        elif spacegeo_biome == 'Subtropical Desert':
            return 'BWh'
    
    # Subtropical Zone (23.5-35°)
    elif 23.5 <= abs_lat < 35:
        if pgen_biome == 'Xeric Shrubland':
            return 'BSh' if elevation < 0.5 else 'BSk'
        elif spacegeo_biome == 'Temperate Seasonal Forest':
            return 'Cfa'
        elif pgen_biome == 'Grasslands':
            return 'BSk'
    
    # Temperate Zone (35-55°)
    elif 35 <= abs_lat < 55:
        if spacegeo_biome == 'Temperate Rainforest':
            return 'Cfb'
        elif pgen_biome == 'Temperate Forest':
            return 'Cfb' if elevation < 0.6 else 'Dfb'
        elif pgen_biome == 'Taiga / Boreal Forest':
            return 'Dfc'
    
    # Boreal Zone (55-66.5°)
    elif 55 <= abs_lat < 66.5:
        if pgen_biome == 'Taiga / Boreal Forest':
            return 'Dfc'
        elif pgen_biome == 'Tundra':
            return 'ET'
    
    # Arctic Zone (>66.5°)
    else:
        return 'EF' if pgen_biome == 'Ice' else 'ET'
    
    # ===== Fallback Rules =====
    # Use spacegeo biomes when more specific
    if spacegeo_biome in BIOME_TO_KOPPEN:
        return BIOME_TO_KOPPEN[spacegeo_biome]
    
    # Default to pgen biome mapping
    return BIOME_TO_KOPPEN.get(pgen_biome, 'Cfb')

def classify_koppen4(pgen_pixel, spacegeo_pixel, elevation, lat_norm):
    """Improved Köppen classification based on actual biome combinations."""
    # Skip ocean pixels
    if tuple(spacegeo_pixel) == (76, 102, 178):
        return 'Ocean'
    
    abs_lat = abs(lat_norm)  # 0=equator, 1=pole
    
    # Get biomes from both maps
    pgen_biome = get_biome(pgen_pixel, PGEN_COLORS)
    spacegeo_biome = get_biome(spacegeo_pixel, SPACEGEO_COLORS)
    
    # ===== Elevation Overrides =====
    if elevation > 0.8:  # High mountains
        return 'EF' if (elevation > 0.9 or pgen_biome == 'Ice') else 'ET'
    
    # ===== Handle Common Problematic Combinations =====
    # 1. Tropical Dry Forest + Tundra (608k cases)
    if (pgen_biome == 'Tropical Dry Forest' and 
        spacegeo_biome == 'Tundra'):
        return 'Cwa' if abs_lat < 0.4 else 'Dfb'  # Force temperate climate
    
    # 2. Taiga/Tundra and Temperate/Tundra transitions (very common)
    if spacegeo_biome == 'Tundra':
        if pgen_biome in ['Taiga / Boreal Forest', 'Temperate Forest']:
            return 'ET' if abs_lat > 0.6 else 'Dfc'  # Arctic or subarctic
    
    # 3. Xeric Shrubland combinations (arid transitions)
    if pgen_biome == 'Xeric Shrubland':
        if spacegeo_biome == 'Savanna':
            return 'BSh' if abs_lat < 0.3 else 'BSk'
        elif spacegeo_biome == 'Subtropical Desert':
            return 'BWh' if abs_lat < 0.35 else 'BWk'
    
    # ===== Primary Biome Classification =====
    # Cases where spacegeo has more specific info
    if spacegeo_biome == 'Tropical Rainforest':
        return 'Af' if abs_lat < 0.3 else 'Cfb'
    elif spacegeo_biome == 'Temperate Rainforest':
        return 'Cfb'
    elif spacegeo_biome == 'Temperate Seasonal Forest':
        return 'Cfa'
    
    # Handle pgen biomes when spacegeo is less specific
    if pgen_biome == 'Tropical Rainforest':
        # Handle edge cases (42 occurrences with Temperate Grassland)
        if spacegeo_biome == 'Temperate Grassland':
            return 'Cwa'  # Monsoon-influenced transition
        return 'Af'
    
    # ===== Fallback Rules =====
    # Use BIOME_TO_KOPPEN but with latitude adjustments
    final_biome = (spacegeo_biome if spacegeo_biome in 
                  ['Tropical Rainforest', 'Temperate Rainforest'] 
                  else pgen_biome)
    
    koppen = BIOME_TO_KOPPEN.get(final_biome, 'Cfb')
    
    # Latitude-based adjustments
    if abs_lat < 0.25:  # Tropical
        if koppen in ['Dfc', 'ET']:  # Prevent arctic in tropics
            return 'Aw'
    elif 0.25 <= abs_lat < 0.4:  # Subtropical
        if koppen == 'Af':  # Limit pure tropical
            return 'Am'
    elif 0.4 <= abs_lat < 0.6:  # Temperate
        if koppen == 'Aw':
            return 'Cwa'  # Transition savanna to monsoon
    
    return koppen

def classify_koppen3(pgen_pixel, spacegeo_pixel, elevation, lat_norm):
    """Improved Köppen classification with better biome handling."""
    # Skip ocean pixels
    if tuple(spacegeo_pixel) == (76, 102, 178):
        return 'Ocean'
    
    abs_lat = abs(lat_norm)  # 0=equator, 1=pole
    
    # Get biomes from both maps
    pgen_biome = get_biome(pgen_pixel, PGEN_COLORS)
    spacegeo_biome = get_biome(spacegeo_pixel, SPACEGEO_COLORS)
    
    # ===== Elevation Overrides =====
    if elevation > 0.8:  # High mountains
        return 'EF' if (elevation > 0.9 or pgen_biome == 'Ice') else 'ET'
    
    # ===== Special Biome Combinations =====
    # Handle Tropical Dry Forest + Temperate Seasonal Forest case
    if (pgen_biome == 'Tropical Dry Forest' and 
        spacegeo_biome == 'Temperate Seasonal Forest'):
        if abs_lat < 0.4:  # Lower latitudes
            return 'Aw' if elevation < 0.5 else 'Cwa'
        else:  # Mid-latitudes (Ferrel cell)
            return 'Cfa' if elevation < 0.6 else 'Cfb'
    
    # ===== Primary Biome Classification =====
    # Handle cases where spacegeo has more specific biome info
    if spacegeo_biome == 'Tropical Rainforest':
        return 'Af' if abs_lat < 0.3 else 'Cfb'
    elif spacegeo_biome == 'Tropical Dry Forest':
        return 'Am' if abs_lat < 0.15 else 'Aw'
    elif spacegeo_biome == 'Savanna':
        return 'Aw'
    elif spacegeo_biome == 'Subtropical Desert':
        return 'BWh'
    
    # Handle pgen biomes
    if pgen_biome == 'Savanna':
        return 'Aw'
    elif pgen_biome == 'Grasslands':
        if abs_lat < 0.3:
            return 'Aw'  # Tropical grassland -> savanna
        else:
            return 'BSk'  # Temperate grassland -> cold steppe
    elif pgen_biome == 'Desert':
        if abs_lat < 0.35:
            return 'BWh'  # Hot desert in tropics/subtropics
        else:
            return 'BWk'  # Cold desert in temperate zones
    elif pgen_biome == 'Xeric Shrubland':
        return 'BSh'
    elif pgen_biome == 'Tropical Dry Forest':
        return 'Aw' if abs_lat < 0.4 else 'Cwa'  # Modified transition
    elif pgen_biome == 'Temperate Forest':
        return 'Cfb' if abs_lat < 0.5 else 'Dfb'
    elif pgen_biome == 'Temperate Rainforest':
        return 'Cfb'
    elif pgen_biome == 'Taiga / Boreal Forest':
        return 'Dfc'
    elif pgen_biome == 'Tundra':
        return 'ET'
    elif pgen_biome == 'Ice':
        return 'EF'
    
    # ===== Fallback Rules =====
    # If we still haven't classified, use latitude-based defaults
    if abs_lat < 0.25:  # Tropical
        return 'Aw'  # Default to savanna in tropics
    elif abs_lat < 0.45:  # Subtropical
        return 'Cfa'  # Default to humid subtropical
    elif abs_lat < 0.65:  # Temperate
        return 'Dfb'  # Default to warm-summer humid continental
    else:  # Polar
        return 'ET'  # Default to tundra

def classify_koppen2(pgen_pixel, spacegeo_pixel, elevation, lat_norm):
    """Improved Köppen classification with better biome handling."""
    # Skip ocean pixels
    if tuple(spacegeo_pixel) == (76, 102, 178):
        return 'Ocean'
    
    abs_lat = abs(lat_norm)  # 0=equator, 1=pole
    
    # Get biomes from both maps
    pgen_biome = get_biome(pgen_pixel, PGEN_COLORS)
    spacegeo_biome = get_biome(spacegeo_pixel, SPACEGEO_COLORS)
    
    # ===== Elevation Overrides =====
    if elevation > 0.8:  # High mountains
        return 'EF' if (elevation > 0.9 or pgen_biome == 'Ice') else 'ET'
    
    # ===== Primary Biome Classification =====
    # Handle cases where spacegeo has more specific biome info
    if spacegeo_biome == 'Tropical Rainforest':
        return 'Af' if abs_lat < 0.3 else 'Cfb'
    elif spacegeo_biome == 'Tropical Dry Forest':
        return 'Am' if abs_lat < 0.15 else 'Aw'
    elif spacegeo_biome == 'Savanna':
        return 'Aw'
    elif spacegeo_biome == 'Subtropical Desert':
        return 'BWh'
    
    # Handle pgen biomes
    if pgen_biome == 'Savanna':
        return 'Aw'
    elif pgen_biome == 'Grasslands':
        if abs_lat < 0.3:
            return 'Aw'  # Tropical grassland -> savanna
        else:
            return 'BSk'  # Temperate grassland -> cold steppe
    elif pgen_biome == 'Desert':
        if abs_lat < 0.35:
            return 'BWh'  # Hot desert in tropics/subtropics
        else:
            return 'BWk'  # Cold desert in temperate zones
    elif pgen_biome == 'Xeric Shrubland':
        return 'BSh'
    elif pgen_biome == 'Tropical Dry Forest':
        return 'Aw'
    elif pgen_biome == 'Temperate Forest':
        return 'Cfb' if abs_lat < 0.5 else 'Dfb'
    elif pgen_biome == 'Temperate Rainforest':
        return 'Cfb'
    elif pgen_biome == 'Taiga / Boreal Forest':
        return 'Dfc'
    elif pgen_biome == 'Tundra':
        return 'ET'
    elif pgen_biome == 'Ice':
        return 'EF'
    
    # ===== Fallback Rules =====
    # If we still haven't classified, use latitude-based defaults
    if abs_lat < 0.25:  # Tropical
        return 'Aw'  # Default to savanna in tropics
    elif abs_lat < 0.45:  # Subtropical
        return 'Cfa'  # Default to humid subtropical
    elif abs_lat < 0.65:  # Temperate
        return 'Dfb'  # Default to warm-summer humid continental
    else:  # Polar
        return 'ET'  # Default to tundra

def classify_koppen0(pgen_pixel, spacegeo_pixel, elevation, lat_norm):
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

def classify_koppen1(pgen_pixel, spacegeo_pixel, elevation, lat_norm):
    """Simplified Köppen classification using only biome maps and elevation."""
    # Skip ocean pixels
    if tuple(spacegeo_pixel) == (76, 102, 178):
        return 'Ocean'
    
    abs_lat = abs(lat_norm)  # 0=equator, 1=pole
    
    # Get biomes from both maps
    pgen_biome = get_biome(pgen_pixel, PGEN_COLORS)
    spacegeo_biome = get_biome(spacegeo_pixel, SPACEGEO_COLORS)
    
    # ===== Elevation Overrides =====
    if elevation > 0.8:  # High mountains
        return 'EF' if (elevation > 0.9 or pgen_biome == 'Ice') else 'ET'
    
    # ===== Climate Zone Determination =====
    # Tropical Zone (0-20°)
    if abs_lat < 0.22:
        if spacegeo_biome == 'Tropical Rainforest':
            return 'Af'
        elif spacegeo_biome == 'Tropical Dry Forest':
            return 'Am' if abs_lat < 0.15 else 'Aw'
        elif pgen_biome in ['Savanna', 'Grasslands']:
            return 'Aw'
        elif pgen_biome == 'Desert':
            return 'BWh'
    
    # Subtropical Zone (20-35°)
    elif 0.22 <= abs_lat < 0.39:
        if pgen_biome == 'Xeric Shrubland':
            return 'BSh'
        elif spacegeo_biome in ['Temperate Seasonal Forest', 'Temperate Rainforest']:
            return 'Cfa'
        elif pgen_biome == 'Grasslands':
            return 'BSk'
        elif pgen_biome == 'Desert':
            return 'BWh' if elevation < 0.5 else 'BWk'
    
    # Temperate Zone (35-55°)
    elif 0.39 <= abs_lat < 0.61:
        if spacegeo_biome == 'Temperate Rainforest':
            return 'Cfb'
        elif pgen_biome in ['Taiga / Boreal Forest', 'Temperate Forest']:
            return 'Dfa' if abs_lat < 0.5 else 'Dfb'
        elif pgen_biome == 'Desert':
            return 'BWk'
    
    # Boreal Zone (55-66.5°)
    elif 0.61 <= abs_lat < 0.74:
        if pgen_biome == 'Taiga / Boreal Forest':
            return 'Dfc'
        elif pgen_biome == 'Tundra':
            return 'ET'
    
    # Polar Zone (>66.5°)
    else:
        return 'EF' if pgen_biome == 'Ice' else 'ET'
    
    # ===== Fallback Classification =====
    return BIOME_TO_KOPPEN.get(
        spacegeo_biome if spacegeo_biome in ['Tropical Rainforest', 'Temperate Rainforest'] 
        else pgen_biome, 
        'Cfb'  # Default to oceanic climate
    )

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

def analyze_biome_combinations(pgen_path, spacegeo_path, output_file=None):
    """
    Analyze and report all unique pgen-spacegeo biome combinations in the maps.
    Optionally saves results to a text file.
    """
    print("\nLoading maps for biome combination analysis...")
    pgen = np.array(Image.open(pgen_path)).astype(np.uint8)
    spacegeo = np.array(Image.open(spacegeo_path).convert('RGB')).astype(np.uint8)
    
    # Validate dimensions
    if pgen.shape != spacegeo.shape:
        raise ValueError("pgen and spacegeo maps must have same dimensions")
    
    height, width, _ = pgen.shape
    combinations = set()
    biome_counts = {}
    
    print("\nAnalyzing biome combinations...")
    OCEAN_COLOR = np.array([76, 102, 178], dtype=np.uint8)
    
    for y in tqdm(range(height), desc="Processing rows", unit="row"):
        for x in range(width):
            # Skip ocean pixels
            if np.array_equal(spacegeo[y, x], OCEAN_COLOR):
                continue
                
            # Get biome names
            pgen_biome = get_biome(pgen[y, x], PGEN_COLORS)
            spacegeo_biome = get_biome(spacegeo[y, x], SPACEGEO_COLORS)
            
            # Record combination
            combo = (pgen_biome, spacegeo_biome)
            combinations.add(combo)
            
            # Count occurrences
            biome_counts[combo] = biome_counts.get(combo, 0) + 1
    
    # Sort combinations by frequency (descending)
    sorted_combinations = sorted(combinations, key=lambda x: -biome_counts[x])
    
    # Prepare report
    report_lines = [
        "Unique PGen-SpaceGeo Biome Combinations Analysis",
        "==============================================",
        f"Total unique combinations found: {len(combinations)}",
        f"Total land pixels analyzed: {sum(biome_counts.values())}",
        "\nCombinations (sorted by frequency):",
        "PGen Biome\t\tSpaceGeo Biome\t\tCount"
    ]
    
    for combo in sorted_combinations:
        pgen_b, spacegeo_b = combo
        count = biome_counts[combo]
        report_lines.append(f"{pgen_b.ljust(20)}\t{spacegeo_b.ljust(20)}\t{count}")
    
    report = "\n".join(report_lines)
    print("\n" + report)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\n✓ Analysis saved to {output_file}")
    else:
        print("\nℹ️ No output file specified - results only printed to console")

def analyze_koppen_distributions(koppen_path, heightmap_path, lat_range=(-90, 90), output_file=None):
    """
    Analyze distributions of Köppen classes by altitude and latitude.
    Generates statistics and optionally saves to file.
    
    Args:
        koppen_path: Path to generated Köppen map (RGB)
        heightmap_path: Path to heightmap (grayscale)
        lat_range: Tuple of (min_lat, max_lat)
        output_file: Optional path to save results
    """
    print("\nLoading data for distribution analysis...")
    
    # Load maps
    with tqdm(desc="Loading Köppen map", unit="file") as pbar:
        koppen_img = np.array(Image.open(koppen_path))
        pbar.update()
    
    with tqdm(desc="Loading heightmap", unit="file") as pbar:
        heightmap = np.array(Image.open(heightmap_path).convert('L')) / 255.0
        pbar.update()
    
    # Validate dimensions
    if koppen_img.shape[:2] != heightmap.shape:
        raise ValueError("Köppen map and heightmap must have same dimensions")
    
    height, width = heightmap.shape
    stats = {
        'by_class': {},       # Köppen class statistics
        'altitude_bins': {},  # Distribution by altitude
        'latitude_bins': {}   # Distribution by latitude
    }
    
    # Prepare reverse color mapping
    COLOR_TO_KOPPEN = {v:k for k,v in KOPPEN_COLORS.items()}
    OCEAN_COLOR = np.array([0, 0, 139], dtype=np.uint8)
    
    # Latitude bins
    lat_min, lat_max = lat_range
    y_coords = np.linspace(lat_max, lat_min, height)
    
    # Bin definitions
    ALTITUDE_BINS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    LATITUDE_BINS = [-90, -60, -30, 0, 30, 60, 90]
    
    print("\nAnalyzing distributions...")
    
    for y in tqdm(range(height), desc="Processing rows", unit="row"):
        current_lat = y_coords[y]
        lat_bin = np.digitize(current_lat, LATITUDE_BINS) - 1  # 0-based
        
        for x in range(width):
            # Skip ocean pixels
            if np.array_equal(koppen_img[y, x], OCEAN_COLOR):
                continue
                
            # Get Köppen class
            koppen_class = COLOR_TO_KOPPEN.get(tuple(koppen_img[y, x]), 'Unknown')
            
            # Get elevation (0-1)
            elevation = heightmap[y, x]
            alt_bin = np.digitize(elevation, ALTITUDE_BINS) - 1  # 0-based
            
            # Initialize data structures
            if koppen_class not in stats['by_class']:
                stats['by_class'][koppen_class] = {
                    'count': 0,
                    'elevation_sum': 0,
                    'lat_sum': 0,
                    'min_alt': 1,
                    'max_alt': 0
                }
            
            # Update class statistics
            stats['by_class'][koppen_class]['count'] += 1
            stats['by_class'][koppen_class]['elevation_sum'] += elevation
            stats['by_class'][koppen_class]['lat_sum'] += current_lat
            stats['by_class'][koppen_class]['min_alt'] = min(
                stats['by_class'][koppen_class]['min_alt'], elevation)
            stats['by_class'][koppen_class]['max_alt'] = max(
                stats['by_class'][koppen_class]['max_alt'], elevation)
            
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
    
    # Calculate averages
    for koppen_class in stats['by_class']:
        count = stats['by_class'][koppen_class]['count']
        stats['by_class'][koppen_class]['mean_alt'] = (
            stats['by_class'][koppen_class]['elevation_sum'] / count)
        stats['by_class'][koppen_class]['mean_lat'] = (
            stats['by_class'][koppen_class]['lat_sum'] / count)
    
    # Generate report
    report_lines = [
        "Köppen Climate Distribution Analysis",
        "=================================",
        f"Total land pixels analyzed: {sum(c['count'] for c in stats['by_class'].values())}",
        "\nOverall Statistics by Climate Class:",
        "Class\tCount\t%Total\tMeanAlt\tMinAlt\tMaxAlt\tMeanLat"
    ]
    
    total_pixels = sum(c['count'] for c in stats['by_class'].values())
    for koppen_class in sorted(stats['by_class'], 
                             key=lambda x: -stats['by_class'][x]['count']):
        data = stats['by_class'][koppen_class]
        pct = (data['count'] / total_pixels) * 100
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
    parser.add_argument("--analyze-distributions", action='store_true',
                      help="Analyze Köppen distributions by altitude/latitude")
    parser.add_argument("--dist-output", default=FPATH_AN_RESULT_DISTRIBUTIONS,
                      help="Output file for distribution analysis")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_biome_combinations(
            args.pgen, args.spacegeo, args.analysis_output
        )
    elif args.analyze_distributions:
        analyze_koppen_distributions(
            args.output, args.heightmap, args.lat, args.dist_output
        )
    else:
        generate_koppen_map(
            args.pgen, args.spacegeo, args.heightmap, 
            args.output, args.lat
        )