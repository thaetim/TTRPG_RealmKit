import numpy as np
from PIL import Image
from pathlib import Path

def combine_heightmaps(linear_path, nonlinear_path, output_path):
    """
    Combine linear and non-linear heightmaps using custom logic.
    
    Args:
        linear_path: Path to linear grayscale heightmap
        nonlinear_path: Path to non-linear grayscale heightmap
        output_path: Path to save the combined heightmap
    """
    # Load images and convert to numpy arrays
    linear = np.array(Image.open(linear_path).convert('L'))
    nonlinear = np.array(Image.open(nonlinear_path).convert('L'))
    
    # Validate dimensions
    if linear.shape != nonlinear.shape:
        raise ValueError("Heightmaps must have identical dimensions")
    
    # Normalize to 0-1 range
    lin_norm = linear.astype(float) / 255.0
    nlin_norm = nonlinear.astype(float) / 255.0
    
    # Apply combination logic (scaled to 0-255)
    # The logic is equivalent to: min(linear, nonlinear) when linear > nonlinear
    #                            max(linear, nonlinear) when linear < nonlinear
    #                            linear when equal
    result = np.where(lin_norm > nlin_norm, 
                     np.minimum(lin_norm, nlin_norm),
                     np.maximum(lin_norm, nlin_norm))
    
    # Convert back to 0-255 range
    result = (result * 255).astype(np.uint8)
    
    # Save the output
    Image.fromarray(result).save(output_path)
    print(f"Combined heightmap saved to {output_path}")

if __name__ == "__main__":
    input_dir = Path(r"D:\DND\Realistic DND World Gen\renders")
    output_dir = Path(r"D:\DND\Realistic DND World Gen\renders")
    
    linear_path = input_dir / "greyscale linear.bmp"
    nonlinear_path = input_dir / "greyscale non-linear.bmp"
    output_path = output_dir / "greyscale combined.bmp"
    
    combine_heightmaps(linear_path, nonlinear_path, output_path)