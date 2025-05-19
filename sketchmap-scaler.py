import argparse
import math
from collections import defaultdict

# Elevation character hierarchy (from lowest to highest)
CHAR_HIERARCHY = [' ', '.', ',', ':', ';', '-', '*', 'o', 'O', '@']

def read_ascii_map(file_path):
    """Reads the ASCII map file into a 2D list."""
    with open(file_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f if line.strip()]
        # Validate uniform line length
        if len(set(len(line) for line in lines)) > 1:
            raise ValueError("All lines in input file must be of equal length")
        return [list(line) for line in lines]

def write_ascii_map(map_data, file_path):
    """Writes a 2D list back to an ASCII map file."""
    with open(file_path, 'w') as f:
        for row in map_data:
            f.write(''.join(row) + '\n')

def get_weighted_char(chars, weights, delta=0.1):
    """
    Returns a character based on weights and delta parameter.
    For midpoint displacement, we want to:
    1. Strongly prefer the dominant character when delta is small
    2. Allow more blending when delta is large
    """
    char_values = defaultdict(float)
    total_weight = sum(weights)
    
    for char, weight in zip(chars, weights):
        # The influence of each character depends on delta
        # Smaller delta = stronger preference for dominant character
        influence = weight * (1.0 / (delta + 0.01))
        char_values[char] += influence
    
    # Normalize and select character
    max_value = max(char_values.values())
    candidates = [c for c, v in char_values.items() if v >= max_value * 0.9]  # Allow some tolerance
    
    # Among candidates, pick the highest in hierarchy
    return max(candidates, key=lambda c: CHAR_HIERARCHY.index(c))

def scale_map(map_data, scale_factor=4, delta=0.1, mode='displacement'):
    """
    Scales up the ASCII map with special consideration for midpoint displacement usage.
    Modes:
    - 'displacement': Optimized for midpoint displacement (default)
    - 'sharp': Sharp transitions (for small delta)
    - 'smooth': Smooth blending (for large delta)
    """
    if not map_data:
        return []
    
    height = len(map_data)
    width = len(map_data[0]) if height > 0 else 0
    scaled_data = []

    for y in range(height * scale_factor):
        original_y = y / scale_factor
        y1 = int(math.floor(original_y))
        y2 = min(int(math.ceil(original_y)), height - 1)
        y_ratio = original_y - y1
        
        row = []
        for x in range(width * scale_factor):
            original_x = x / scale_factor
            x1 = int(math.floor(original_x))
            x2 = min(int(math.ceil(original_x)), width - 1)
            x_ratio = original_x - x1

            if mode == 'sharp' or delta < 0.05:
                # For very small delta or sharp mode, use nearest neighbor
                chosen_char = map_data[y1][x1]
            else:
                # Get the 4 nearest characters
                tl = map_data[y1][x1]
                tr = map_data[y1][x2]
                bl = map_data[y2][x1]
                br = map_data[y2][x2]

                # Calculate weights
                weights = [
                    (1 - x_ratio) * (1 - y_ratio),  # Top-left
                    x_ratio * (1 - y_ratio),         # Top-right
                    (1 - x_ratio) * y_ratio,         # Bottom-left
                    x_ratio * y_ratio                # Bottom-right
                ]

                if mode == 'displacement':
                    chosen_char = get_weighted_char([tl, tr, bl, br], weights, delta)
                else:  # smooth mode
                    # For large delta, allow more blending
                    chosen_char = get_weighted_char([tl, tr, bl, br], weights, delta=0.3)
            
            row.append(chosen_char)
        scaled_data.append(row)

    return scaled_data

def main():
    parser = argparse.ArgumentParser(description='Scale up an ASCII elevation map for terrain generation.')
    parser.add_argument('input', help='Input map file (e.g., sketch_v7.map)')
    parser.add_argument('output', help='Output map file (e.g., sketch.map)')
    parser.add_argument('--scale', type=int, default=4,
                      help='Scaling factor (default: 4)')
    parser.add_argument('--delta', type=float, default=0.1,
                      help='Delta parameter for terrain generation (0.03-1.0, default: 0.1)')
    parser.add_argument('--mode', choices=['displacement', 'sharp', 'smooth'], default='displacement',
                      help='Interpolation mode (default: displacement)')
    args = parser.parse_args()

    # Validate arguments
    if args.scale < 1:
        raise ValueError("Scale factor must be at least 1")
    if not 0.03 <= args.delta <= 1.0:
        raise ValueError("Delta should be between 0.03 and 1.0")

    original_map = read_ascii_map(args.input)
    scaled_map = scale_map(
        original_map,
        scale_factor=args.scale,
        delta=args.delta,
        mode=args.mode
    )
    write_ascii_map(scaled_map, args.output)

    print(f"Successfully scaled {args.input} -> {args.output} ({args.scale}x)")
    print(f"Mode: {args.mode}, Delta: {args.delta}")

if __name__ == '__main__':
    main()