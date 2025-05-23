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

KOPPEN_TO_COLOR = {
    'Af': (0, 0, 254),     # Tropical rainforest - Dark blue
    'Am': (0, 119, 255),   # Tropical monsoon - Medium blue
    'Aw': (70, 169, 250),   # Tropical savanna - Light blue
    'As': (121, 186, 236),  # Tropical dry summer - Very light blue
    'BWh': (254, 0, 0),     # Hot desert - Red
    'BWk': (254, 150, 149),  # Cold desert - Pink
    'BSh': (245, 163, 1),   # Hot semi-arid - Orange
    'BSk': (255, 219, 99),  # Cold semi-arid - Light yellow
    'Csa': (255, 255, 0),   # Hot-summer Mediterranean - Yellow
    'Csb': (198, 199, 0),   # Warm-summer Mediterranean - Olive
    'Csc': (150, 150, 0),   # Cold-summer Mediterranean - Dark olive
    # Monsoon-influenced humid subtropical - Light green
    'Cwa': (150, 255, 150),
    'Cwb': (99, 199, 100),  # Subtropical highland - Medium green
    'Cwc': (50, 150, 50),   # Cold subtropical highland - Dark green
    'Cfa': (198, 255, 78),  # Humid subtropical - Bright green
    'Cfb': (102, 255, 51),  # Oceanic - Lime green
    'Cfc': (51, 199, 1),    # Subpolar oceanic - Forest green
    'Dsa': (255, 0, 254),   # Hot-summer humid continental - Magenta
    'Dsb': (198, 0, 199),   # Warm-summer humid continental - Purple
    'Dsc': (150, 50, 149),  # Cold-summer humid continental - Dark purple
    # Very cold winter humid continental - Very dark purple
    'Dsd': (150, 100, 149),
    # Monsoon-influenced hot-summer humid continental - Light blue-purple
    'Dwa': (171, 177, 255),
    # Monsoon-influenced warm-summer humid continental - Medium blue-purple
    'Dwb': (90, 119, 219),
    'Dwc': (76, 81, 181),   # Monsoon-influenced subarctic - Dark blue-purple
    # Monsoon-influenced extremely cold subarctic - Very dark blue
    'Dwd': (50, 0, 135),
    'Dfa': (0, 255, 255),   # Hot-summer humid continental - Cyan
    'Dfb': (56, 199, 255),  # Warm-summer humid continental - Sky blue
    'Dfc': (0, 126, 125),   # Subarctic - Teal
    'Dfd': (0, 69, 94),     # Extremely cold subarctic - Dark teal
    'ET': (178, 178, 178),  # Tundra - Light gray
    'EF': (104, 104, 104),  # Ice cap - Dark gray
    'Ocean': (0, 0, 139)    # Deep ocean blue (added for water)
}

# Precomputed constants and lookup tables
OCEAN_COLOR = np.array([76, 102, 178], dtype=np.uint8)
COLOR_TO_KOPPEN = {tuple(v): k for k, v in KOPPEN_TO_COLOR.items()}
