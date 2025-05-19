import numpy as np
from PIL import Image
from pathlib import Path

def combine_heightmaps(linear_path, nonlinear_path, output_path):
    """
    Combine heightmaps using a mathematical formula that replicates the truth table.
    Formula: result = (linear + nonlinear - abs(linear - nonlinear)) / 2
    """
    # Load images
    linear = np.array(Image.open(linear_path).convert('L'))
    nonlinear = np.array(Image.open(nonlinear_path).convert('L'))
    
    # Validate dimensions
    if linear.shape != nonlinear.shape:
        raise ValueError("Heightmaps must have identical dimensions")
    
    # Convert to float for calculations
    a = linear.astype(float)
    b = nonlinear.astype(float)

    # Calculate using a formula
    # result = (lin + nlin - np.abs(lin - nlin)) / 2
    # result = np.where(lin > nlin, nlin, np.where(lin < nlin, nlin, lin))
    # result = (a + b - np.abs(a - b)) / 2

    # For 0-255 scale implementation
    m = 127 # midpoint
    result = a * np.abs(b - m)/m + b * (1 - np.abs(b - m)/m)
    
    # Convert back to 8-bit
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Save output
    Image.fromarray(result).save(output_path)
    print(f"Combined heightmap saved to {output_path}")

# Test with your example values (0-2 scale)
def test_formula():
    linear = np.array([2,2,2,1,1,1,0,0,0])
    nonlinear = np.array([2,1,0,2,1,0,2,1,0])
    
    # Apply formula
    result = (linear + nonlinear - np.abs(linear - nonlinear)) / 2
    
    print("Linear:   ", linear)
    print("Nonlinear:", nonlinear)
    print("Result:   ", result)
    print("Matches expected output:", 
          np.array_equal(result, [2,1,2,1,1,1,0,1,0]))

if __name__ == "__main__":
    test_formula()  # Verify with test case
    
    input_dir = Path(r"D:\DND\Realistic DND World Gen\renders")
    output_dir = input_dir
    
    linear_path = input_dir / "greyscale linear.bmp"
    nonlinear_path = input_dir / "greyscale non-linear.bmp"
    output_path = output_dir / "greyscale combined.bmp"
    
    combine_heightmaps(linear_path, nonlinear_path, output_path)