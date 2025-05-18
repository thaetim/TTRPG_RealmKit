import argparse
from collections import defaultdict

# Elevation character hierarchy (from lowest to highest)
CHAR_HIERARCHY = ['.', ',', ':', ';', '-', '*', 'o', 'O', '@']

def read_ascii_map(file_path):
    """Reads the ASCII map file into a 2D list."""
    with open(file_path, 'r') as f:
        return [list(line.rstrip('\n')) for line in f if line.strip()]

def write_ascii_map(map_data, file_path):
    """Writes a 2D list back to an ASCII map file."""
    with open(file_path, 'w') as f:
        for row in map_data:
            f.write(''.join(row) + '\n')

def get_dominant_char(chars, weights, strength=1.0):
    """
    Returns the highest elevation character weighted by distance.
    Strength controls how aggressively higher elevations dominate:
    0.0 = equal weighting
    1.0 = normal hierarchy weighting
    >1.0 = stronger preference for higher elevations
    """
    char_values = defaultdict(float)
    for char, weight in zip(chars, weights):
        # Apply strength factor to hierarchy position
        hierarchy_weight = (CHAR_HIERARCHY.index(char) / len(CHAR_HIERARCHY)) ** strength
        char_values[char] += weight * (1 + hierarchy_weight)
    
    return max(char_values.items(), key=lambda x: x[1])[0]

def scale_map(map_data, scale_factor=4, interpolate=True, interpolation_strength=1.0):
    """Scales up the ASCII map with elevation-aware interpolation."""
    if not map_data:
        return []
    
    height = len(map_data)
    width = len(map_data[0]) if height > 0 else 0
    scaled_data = []

    for y in range(height * scale_factor):
        original_y = y // scale_factor
        next_y = min(original_y + 1, height - 1)
        y_ratio = (y % scale_factor) / scale_factor
        row = []

        for x in range(width * scale_factor):
            original_x = x // scale_factor
            next_x = min(original_x + 1, width - 1)
            x_ratio = (x % scale_factor) / scale_factor

            if not interpolate:
                # Pixel-perfect scaling
                row.append(map_data[original_y][original_x])
            else:
                # Get 4 nearest characters
                tl = map_data[original_y][original_x]
                tr = map_data[original_y][next_x]
                bl = map_data[next_y][original_x]
                br = map_data[next_y][next_x]

                # Calculate weights
                weights = [
                    (1 - x_ratio) * (1 - y_ratio),  # Top-left
                    x_ratio * (1 - y_ratio),         # Top-right
                    (1 - x_ratio) * y_ratio,         # Bottom-left
                    x_ratio * y_ratio                # Bottom-right
                ]

                # Choose dominant character with strength factor
                dominant_char = get_dominant_char([tl, tr, bl, br], weights, interpolation_strength)
                row.append(dominant_char)

        scaled_data.append(row)

    return scaled_data

def main():
    parser = argparse.ArgumentParser(description='Scale up an ASCII elevation map.')
    parser.add_argument('input', help='Input map file (e.g., sketch_v7.map)')
    parser.add_argument('output', help='Output map file (e.g., sketch.map)')
    parser.add_argument('--scale', type=int, default=4,
                      help='Scaling factor (default: 4)')
    parser.add_argument('--pixel-perfect', action='store_true',
                      help='Disable interpolation (strict pixel scaling)')
    parser.add_argument('--interp-strength', type=float, default=1.0,
                      help='Interpolation strength (0.0-2.0, default: 1.0)')
    args = parser.parse_args()

    # Validate arguments
    if args.scale < 1:
        raise ValueError("Scale factor must be at least 1")
    if args.interp_strength < 0 or args.interp_strength > 2:
        raise ValueError("Interpolation strength should be between 0.0 and 2.0")

    original_map = read_ascii_map(args.input)
    scaled_map = scale_map(
        original_map,
        scale_factor=args.scale,
        interpolate=not args.pixel_perfect,
        interpolation_strength=args.interp_strength
    )
    write_ascii_map(scaled_map, args.output)

    print(f"Successfully scaled {args.input} -> {args.output} ({args.scale}x)")
    if not args.pixel_perfect:
        print(f"Interpolation strength: {args.interp_strength:.1f}")
    else:
        print("Mode: Pixel-perfect")

if __name__ == '__main__':
    main()