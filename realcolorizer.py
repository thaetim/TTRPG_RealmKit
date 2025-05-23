import numba
import time
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
from skimage import color
from numba import jit, prange

from biome_arrange_constants import *

import os
os.environ["OPENBLAS_NUM_THREADS"] = "3"


# Numerical month mapping
MONTH_TO_NUM = {'jan': 0, 'apr': 1, 'jul': 2, 'oct': 3}
NUM_TO_MONTH = ['jan', 'apr', 'jul', 'oct']

# Numba-optimized color space conversions


@numba.njit
def rgb_to_hsv_numba(rgb):
    """Convert RGB to HSV for Nx3 array in 0-1 range."""
    hsv = np.empty_like(rgb)
    for i in range(rgb.shape[0]):
        r = rgb[i, 0]
        g = rgb[i, 1]
        b = rgb[i, 2]

        maxc = max(r, g, b)
        minc = min(r, g, b)
        v = maxc

        if maxc == 0:
            s = 0.0
        else:
            s = (maxc - minc) / maxc

        if maxc == minc:
            h = 0.0
        else:
            rc = (maxc - r) / (maxc - minc + 1e-10)
            gc = (maxc - g) / (maxc - minc + 1e-10)
            bc = (maxc - b) / (maxc - minc + 1e-10)

            if r == maxc:
                h = bc - gc
            elif g == maxc:
                h = 2.0 + rc - bc
            else:
                h = 4.0 + gc - rc

            h = (h / 6.0) % 1.0

        hsv[i, 0] = h
        hsv[i, 1] = s
        hsv[i, 2] = v
    return hsv


@numba.njit
def hsv_to_rgb_numba(hsv):
    """Convert HSV to RGB for Nx3 array in 0-1 range."""
    rgb = np.empty_like(hsv)
    for i in range(hsv.shape[0]):
        h, s, v = hsv[i, 0], hsv[i, 1], hsv[i, 2]
        if s == 0:
            r = g = b = v
        else:
            h6 = h * 6.0
            i_floor = int(h6)
            f = h6 - i_floor
            p = v * (1.0 - s)
            q = v * (1.0 - s * f)
            t = v * (1.0 - s * (1.0 - f))

            i_floor %= 6
            if i_floor == 0:
                r, g, b = v, t, p
            elif i_floor == 1:
                r, g, b = q, v, p
            elif i_floor == 2:
                r, g, b = p, v, t
            elif i_floor == 3:
                r, g, b = p, q, v
            elif i_floor == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q

        rgb[i, 0] = r
        rgb[i, 1] = g
        rgb[i, 2] = b
    return rgb


def load_color_data(json_path):
    """Load color data with numerical month indexing."""
    with open(json_path) as f:
        data = json.load(f)

    color_data = {}

    if 'monthly_colors' in data:
        for koppen_code, monthly_data in data['monthly_colors'].items():
            color_array = np.zeros((4, 3), dtype=np.float32)
            counts = np.zeros(4, dtype=int)

            for month_str, month_data in monthly_data.items():
                month_num = MONTH_TO_NUM.get(month_str[:3].lower(), -1)
                if month_num == -1:
                    continue

                base_color = np.array(month_data['mean'], dtype=np.float32)
                hsv = rgb_to_hsv_numba(base_color.reshape(1, 3)/255.0)[0]
                hsv[1] = min(1.0, hsv[1]*1.2)
                enhanced_color = hsv_to_rgb_numba(hsv.reshape(1, 3))[0] * 255
                color_array[month_num] = enhanced_color
                counts[month_num] += 1

            # Fill missing months with average of available
            avg_color = np.mean(color_array[counts > 0], axis=0) if np.any(
                counts) else np.array([120.0, 100.0, 60.0])
            for i in range(4):
                if counts[i] == 0:
                    color_array[i] = avg_color

            color_data[koppen_code] = color_array

    return color_data


def apply_realistic_colors_optimized(koppen_map_path, heightmap_path, output_path, month="jul",
                                     json_path=None, add_variation=True, skip_ocean=False):
    """Optimized version using numerical month indexing and Numba."""
    start_time = time.time()

    # Load and preprocess data
    color_data = load_color_data(json_path)
    koppen_img = np.array(Image.open(koppen_map_path))
    heightmap = np.array(Image.open(heightmap_path).convert(
        'L'), dtype=np.float32) / 255.0
    height, width = heightmap.shape
    output = np.zeros((height, width, 3), dtype=np.float32)

    # Precompute numerical month indices
    current_month_num = MONTH_TO_NUM[month.lower()[:3]]
    y_coords = np.arange(height)
    northern_mask = y_coords < height // 2

    # Create month map for each pixel
    month_map = np.zeros((height, width), dtype=np.int32)
    for y in range(height):
        if northern_mask[y]:
            month_map[y, :] = current_month_num
        else:
            month_map[y, :] = (current_month_num + 2) % 4

    # Process ocean first
    if not skip_ocean and OCEAN_COLOR is not None:
        ocean_color = np.array(OCEAN_COLOR, dtype=np.float32)
        ocean_mask = np.all(koppen_img == ocean_color, axis=-1)
        if np.any(ocean_mask):
            depth_factor = np.clip(0.3 + heightmap * 1.5, 0.3, 2.0)
            output[ocean_mask] = np.stack([
                20 * depth_factor[ocean_mask],
                100 * depth_factor[ocean_mask],
                200 * depth_factor[ocean_mask]
            ], axis=1)
        land_mask = ~ocean_mask
    else:
        land_mask = np.ones_like(heightmap, dtype=bool)

    # Process land classes
    for koppen_class, color_array in color_data.items():
        if koppen_class == 'Ocean':
            continue

        class_color = np.array(KOPPEN_TO_COLOR[koppen_class], dtype=np.float32)
        class_mask = np.all(koppen_img == class_color, axis=-1) & land_mask
        if not np.any(class_mask):
            continue

        # Get month indices for masked pixels
        masked_month_indices = month_map[class_mask]
        base_colors = color_array[masked_month_indices]

        # Elevation effects
        heights = heightmap[class_mask]
        elev_factors = 0.85 + heights * 0.3
        shaded_colors = base_colors * elev_factors[:, np.newaxis]

        # High elevation snow effect
        if koppen_class in ['ET', 'EF', 'Dfc', 'Dfd', 'Dsc', 'Dsd']:
            snow_mask = heights > 0.85
            if np.any(snow_mask):
                snow_amount = np.clip((heights[snow_mask] - 0.85) * 6.0, 0, 1)
                shaded_colors[snow_mask] = (
                    shaded_colors[snow_mask] * (1 - snow_amount[:, np.newaxis]) +
                    np.array([230, 240, 255]) * snow_amount[:, np.newaxis]
                )

        # Add variation
        if add_variation:
            noise = np.random.randint(-3, 4,
                                      size=shaded_colors.shape, dtype=np.int32)
            shaded_colors = np.clip(shaded_colors + noise, 0, 255)

        # HSV saturation boost
        hsv_colors = rgb_to_hsv_numba(shaded_colors / 255.0)
        hsv_colors[:, 1] = np.minimum(1.0, hsv_colors[:, 1] * 1.3)
        shaded_colors = hsv_to_rgb_numba(hsv_colors) * 255

        output[class_mask] = np.clip(shaded_colors, 0, 255)

    # Save result
    output_img = output.astype(np.uint8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_img).save(output_path)

    total_time = time.time() - start_time
    print(f"Processed {height}x{width} image in {total_time:.2f}s")
    return output_img


def get_season_by_hemisphere(y, height, month):
    """Adjust season based on hemisphere (y-coordinate)"""
    month = month.lower()[:3]  # Ensure we have 3-letter month abbreviation

    if y < height // 2:  # Northern hemisphere
        return month
    else:  # Southern hemisphere - invert seasons
        season_map = {
            "jan": "jul",
            "feb": "aug",
            "mar": "sep",
            "apr": "oct",
            "may": "nov",
            "jun": "dec",
            "jul": "jan",
            "aug": "feb",
            "sep": "mar",
            "oct": "apr",
            "nov": "may",
            "dec": "jun"
        }
        return season_map.get(month, month)


def enhance_color_realism(colors, month):
    """Apply color theory and natural variation to make colors more realistic"""
    enhanced_colors = {}

    for koppen_code, monthly_colors in colors.items():
        enhanced_colors[koppen_code] = {}

        # Use monthly color if available
        if month in monthly_colors:
            base_color = monthly_colors[month]
        else:
            # Fallback to average of all available months
            all_month_colors = list(monthly_colors.values())
            base_color = np.mean(
                all_month_colors, axis=0) if all_month_colors else np.array([120, 100, 60])

        # Add variation
        variation = np.random.randint(-5, 6, size=3)
        varied_color = np.clip(np.array(base_color) + variation, 0, 255)

        enhanced_colors[koppen_code][month] = tuple(varied_color)

    return enhanced_colors


@jit(nopython=True, parallel=True)
def apply_elevation_effects_vectorized(base_colors, heightmap, class_mask, koppen_class,
                                       low_color, mid_color, high_color, has_low, has_mid, has_high):
    """Vectorized elevation effects using Numba"""
    height, width = heightmap.shape

    # Process each pixel in parallel
    for y in prange(height):
        for x in range(width):
            if not class_mask[y, x]:
                continue

            # Apply elevation colors based on height
            if heightmap[y, x] < 0.3 and has_low:
                base_colors[y, x] = low_color
            elif heightmap[y, x] < 0.7 and has_mid:
                base_colors[y, x] = mid_color
            elif has_high:
                base_colors[y, x] = high_color

            # Special snow effect for cold climates at high elevation
            if heightmap[y, x] >= 0.85:
                is_cold = False
                for climate in ['ET', 'EF', 'Dfc', 'Dfd', 'Dsc', 'Dsd']:
                    if koppen_class == climate:
                        is_cold = True
                        break

                if is_cold:
                    snow_amount = np.clip((heightmap[y, x] - 0.85) * 6.0, 0, 1)
                    snow_color = np.array([230, 240, 255], dtype=np.float32)
                    base_colors[y, x] = (
                        base_colors[y, x] * (1 - snow_amount) +
                        snow_color * snow_amount
                    )

    return base_colors


@jit(nopython=True, parallel=True)
def apply_color_variation(colors, variation_range=3):
    """Apply random color variation using Numba"""
    noise = np.random.randint(-variation_range,
                              variation_range + 1, size=colors.shape)
    return np.clip(colors + noise, 0, 255)


def apply_realistic_colors_fast(koppen_map_path, heightmap_path, output_path, month="jul",
                                json_path=None, add_variation=True, skip_ocean=False):
    """
    Ultra-fast vectorized implementation for realistic biome coloring
    Args:
        koppen_map_path: Path to Köppen climate map image
        heightmap_path: Path to heightmap image
        output_path: Output file path
        month: Target month (jan/apr/jul/oct)
        json_path: Path to color data JSON file
        add_variation: Add random color variation if True
        skip_ocean: Skip ocean processing if True
    """

    # Load color data
    with open(json_path) as f:
        color_data = json.load(f)

    # Extract color information - precompute all arrays
    colors = {}
    if 'monthly_colors' in color_data:
        for koppen_class, monthly_data in color_data['monthly_colors'].items():
            colors[koppen_class] = {}
            for m, month_data in monthly_data.items():
                if 'mean' in month_data:
                    colors[koppen_class][m] = np.array(
                        month_data['mean'], dtype=np.float32)
                if 'elevation_colors' in month_data:
                    colors[koppen_class].setdefault('elevation', {})
                    for elev, color in month_data['elevation_colors'].items():
                        if color is not None:
                            colors[koppen_class]['elevation'][elev] = np.array(
                                color, dtype=np.float32)

    # Load maps - convert to float32 immediately
    koppen_img = np.array(Image.open(koppen_map_path), dtype=np.float32)
    heightmap = np.array(Image.open(heightmap_path).convert(
        'L'), dtype=np.float32) / 255.0
    height, width = heightmap.shape
    total_pixels = height * width

    # Initialize output
    realistic_img = np.zeros((height, width, 3), dtype=np.float32)

    # Precompute coordinate grids and masks
    y_coords = np.arange(height, dtype=np.float32)[:, np.newaxis]
    northern_mask = y_coords < height // 2

    # Season mapping (vectorized) - precompute all month masks
    month = month.lower()[:3]
    season_map = {
        'jan': np.where(northern_mask, 'jan', 'jul'),
        'apr': np.where(northern_mask, 'apr', 'oct'),
        'jul': np.where(northern_mask, 'jul', 'jan'),
        'oct': np.where(northern_mask, 'oct', 'apr')
    }
    current_month_map = season_map[month]

    # Get list of land classes to process
    land_classes = [k for k in colors.keys() if k != 'Ocean' and (
        KOPPEN_TO_COLOR is None or k in KOPPEN_TO_COLOR)]

    # Process ocean first if not skipping
    if not skip_ocean and 'Ocean' in colors and OCEAN_COLOR is not None:
        ocean_color = np.array(OCEAN_COLOR, dtype=np.float32)
        ocean_mask = np.all(koppen_img == ocean_color, axis=-1)
        if np.any(ocean_mask):
            depth_factor = np.maximum(0.3, 1.0 + heightmap * 1.5)
            realistic_img[ocean_mask] = np.dstack([
                20 * depth_factor[ocean_mask],
                100 * depth_factor[ocean_mask],
                200 * depth_factor[ocean_mask]
            ])
        land_mask = ~ocean_mask
    else:
        land_mask = np.ones_like(heightmap, dtype=bool)

    # Process land biomes
    for koppen_class in land_classes:
        # Create class mask
        class_color = np.array(KOPPEN_TO_COLOR[koppen_class], dtype=np.float32)
        class_mask = np.all(koppen_img == class_color, axis=-1) & land_mask

        if not np.any(class_mask):
            continue

        # Initialize base colors array
        base_colors = np.zeros((height, width, 3), dtype=np.float32)

        # Assign colors based on hemisphere-adjusted months
        for m in ['jan', 'apr', 'jul', 'oct']:
            if m not in colors[koppen_class]:
                continue
            month_mask = current_month_map == m
            base_colors[class_mask & month_mask] = colors[koppen_class][m]

        # Apply elevation effects
        if 'elevation' in colors[koppen_class]:
            elev_colors = colors[koppen_class]['elevation']
            low_color = elev_colors.get('low', np.zeros(3, dtype=np.float32))
            mid_color = elev_colors.get('mid', np.zeros(3, dtype=np.float32))
            high_color = elev_colors.get('high', np.zeros(3, dtype=np.float32))

            base_colors = apply_elevation_effects_vectorized(
                base_colors, heightmap, class_mask, koppen_class,
                low_color, mid_color, high_color,
                'low' in elev_colors, 'mid' in elev_colors, 'high' in elev_colors
            )

        # Apply general elevation shading
        elev_factor = 0.85 + (heightmap[class_mask] * 0.3)
        shaded_colors = base_colors[class_mask] * elev_factor[:, np.newaxis]

        # Add variation if enabled
        if add_variation:
            shaded_colors = apply_color_variation(shaded_colors)

        # Apply saturation boost to land areas using scikit-image
        rgb_normalized = shaded_colors / 255.0
        hsv = color.rgb2hsv(rgb_normalized)
        hsv[:, 1] = np.minimum(1.0, hsv[:, 1] * 1.3)
        shaded_colors = color.hsv2rgb(hsv) * 255

        realistic_img[class_mask] = np.clip(shaded_colors, 0, 255)

    # Convert to uint8 only at the end
    realistic_img = realistic_img.astype(np.uint8)

    # Save output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(realistic_img).save(output_path)

    return realistic_img


def generate_all_months(koppen_map_path, heightmap_path, output_dir, add_variation=True, json_path=None, skip_ocean=False):
    """Generate realistic biome maps for all four months with hemisphere awareness"""
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

    # Generate each month
    for month in ["jan", "apr", "jul", "oct"]:
        output_path = output_dir / f"realistic_biomes_{month}.png"
        realistic_img = np.zeros((height, width, 3), dtype=np.uint8)

        missing_classes = set()
        missing_months = set()

        with tqdm(total=height*width, desc=f"Generating {month} map") as pbar:
            for y in range(height):
                # Determine month based on hemisphere
                current_month = get_season_by_hemisphere(y, height, month)

                for x in range(width):
                    # Get Köppen class
                    pixel_tuple = tuple(koppen_img[y, x])
                    koppen_class = COLOR_TO_KOPPEN.get(pixel_tuple, 'Ocean')

                    # Skip ocean processing if requested
                    if skip_ocean and koppen_class == 'Ocean':
                        # Keep original ocean color
                        realistic_img[y, x] = koppen_img[y, x]
                        pbar.update(1)
                        continue

                    # Get base color for this biome + month
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
                            base_color = np.array(
                                colors[koppen_class][current_month])
                        except (KeyError, TypeError) as e:
                            # Track missing data
                            if koppen_class not in colors:
                                missing_classes.add(koppen_class)
                            elif current_month not in colors[koppen_class]:
                                missing_months.add(
                                    f"{koppen_class}:{current_month}")
                            else:
                                missing_classes.add(
                                    f"ERROR:{koppen_class}:{str(e)}")

                            base_color = np.array(
                                [120, 100, 60])  # Fallback color

                    # Apply elevation shading
                    elev_factor = 0.85 + (heightmap[y, x] * 0.3)
                    shaded_color = base_color * elev_factor

                    # Add slight noise for natural variation
                    if koppen_class != 'Ocean' and add_variation:
                        color_noise = np.random.randint(-3, 4, size=3)
                        final_color = np.clip(
                            shaded_color + color_noise, 0, 255)
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
        if missing_classes or missing_months:
            missing_file = output_dir / f"missing_color_data_{month}.txt"
            with open(missing_file, "w") as f:
                if missing_classes:
                    f.write("Missing Köppen classes:\n")
                    f.write("\n".join(sorted(missing_classes)) + "\n\n")
                if missing_months:
                    f.write("Missing month data (format: KöppenClass:Month):\n")
                    f.write("\n".join(sorted(missing_months)) + "\n")

        # Save output
        Image.fromarray(realistic_img).save(output_path)
        print(f"\n✓ Realistic {month} biome map saved to {output_path}")


if __name__ == "__main__":
    # Default file paths
    MAIN_WDIR = Path(r"D:\DND\Realistic DND World Gen")
    DEFAULT_HEIGHTMAP = MAIN_WDIR / "renders" / "greyscale tuned.bmp"
    DEFAULT_KOPPEN = MAIN_WDIR / "climate" / "koppen tuned v2.bmp"
    DEFAULT_OUTPUT = MAIN_WDIR / "climate"
    DEFAULT_CLIMATE_PALETTES = MAIN_WDIR / \
        "climate" / "earth" / "climate_palettes.json"

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
    parser.add_argument("--month", choices=["jan", "apr", "jul", "oct"], default="jul",
                        help="Which month to generate (if not generating all)")
    parser.add_argument("--all-months", action="store_true",
                        help="Generate maps for all four months")
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

    if args.all_months:
        generate_all_months(
            args.koppen,
            args.heightmap,
            output_dir,
            add_variation=not args.no_variation,
            json_path=args.json,
            skip_ocean=args.skip_ocean
        )
    else:
        output_path = output_dir / f"realistic_biomes_{args.month}.png"
        apply_realistic_colors_optimized(
            args.koppen,
            args.heightmap,
            output_path,
            month=args.month,
            add_variation=not args.no_variation,
            json_path=args.json,
            skip_ocean=args.skip_ocean
        )
