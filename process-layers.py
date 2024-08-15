# type: ignore
from gimpfu import *
import os
import sys

DIRPATH_RENDERS = r"H:\DND\Realistic DND World Gen\renders"


def load_images_as_layers(image, directory_path):
    """Load all images from the specified directory as separate layers in the GIMP project."""
    layers = []
    if not os.path.isdir(directory_path):
        pdb.gimp_message(f"Directory does not exist: {directory_path}")
        return layers

    # List all files in the directory
    filepaths = [os.path.join(directory_path, f) for f in os.listdir(
        directory_path) if f.endswith('.png')]

    if not filepaths:
        pdb.gimp_message(f"No PNG files found in directory: {directory_path}")

    # Sort filepaths if necessary (e.g., by specific naming conventions)
    filepaths.sort()

    for filepath in filepaths:
        if not os.path.isfile(filepath):
            pdb.gimp_message(f"File does not exist: {filepath}")
            continue

        # Load each image file as a layer
        layer = pdb.gimp_file_load_layer(image, filepath)
        pdb.gimp_image_add_layer(image, layer, 0)
        layers.append(layer)

    return layers


def select_landmasses(bathy_layer):
    """Select the landmasses based on bathymetric coloring."""
    # Assuming the land starts at color index 131 and above
    pdb.gimp_image_select_color_range(
        bathy_layer.image, CHANNEL_OP_REPLACE, bathy_layer, 131)


def create_erase_layer(image, name, selection_mask):
    """Create a new layer with selected areas filled with black."""
    erase_layer = pdb.gimp_layer_new(
        image, image.width, image.height, RGBA_IMAGE, name, 100, NORMAL_MODE)
    pdb.gimp_image_add_layer(image, erase_layer, 0)

    # Set the layer to be transparent initially
    pdb.gimp_drawable_fill(erase_layer, TRANSPARENT_FILL)

    # Fill the selection with black
    pdb.gimp_edit_fill(erase_layer, selection_mask)

    return erase_layer


def main():
    """Main function to process images and create erase layers."""
    directory_path = DIRPATH_RENDERS

    # Create a new image in GIMP
    image = pdb.gimp_image_new(7500, 3750, RGB)

    # Load the images as layers
    layers = load_images_as_layers(image, directory_path)

    if not layers:
        pdb.gimp_message("No layers loaded. Exiting.")
        return

    # Assuming the bathymetric layer is the first one
    bathy_layer = layers[0]

    # Select the landmasses on the bathymetric layer
    select_landmasses(bathy_layer)

    # Create land_erase layer
    land_erase_layer = create_erase_layer(image, "land_erase", FOREGROUND_FILL)

    # Invert the selection to select the oceans
    pdb.gimp_selection_invert(image)

    # Create ocean_erase layer
    ocean_erase_layer = create_erase_layer(
        image, "ocean_erase", FOREGROUND_FILL)

    # Display the image in GIMP
    pdb.gimp_display_new(image)


# Get directory path from arguments or set default path
if __name__ == "__main__":
    if len(sys.argv) > 1:
        render_path = sys.argv[1]
    else:
        print("No directory path specified. Exiting.")
        sys.exit(1)

    main(render_path)

register(
    "python-fu-process-maps",
    "Process maps to create land and ocean erase layers",
    "Loads generated images, selects landmasses, and creates erase layers",
    "Your Name", "Your Name", "2024",
    "<Image>/Filters/Custom/Process Maps",
    "",
    [],
    [],
    main)
