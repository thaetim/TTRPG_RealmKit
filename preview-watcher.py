import os
import subprocess
import time
from pathlib import Path


def run_command(show_output=True):
    # Define the paths
    planet_dir = Path('planet')
    root_dir = Path(__file__).parent.resolve()  # Get the root directory
    sketch_file = root_dir / 'sketch.map'
    output_file = root_dir / 'preview.bmp'

    # Command to run
    command = f'planet.exe -s 0.2 -i -0.042 -w 1024 -h 512 -o "{
        output_file}" -m 1 -p q -C olssonlight.col -z -V -0.069 -v -0.3721 -M 0.1 < "{sketch_file}"'

    # Determine the output settings
    stdout_setting = None if show_output else subprocess.DEVNULL
    stderr_setting = None if show_output else subprocess.DEVNULL

    # Change the working directory and run the command
    subprocess.run(command, shell=True, cwd=planet_dir,
                   stdout=stdout_setting, stderr=stderr_setting)


def update_html_with_map_sketch(html_file, sketch_file):
    with open(sketch_file, 'r') as sketch:
        map_sketch_content = sketch.read()

    # Update the HTML file with the new map sketch content
    with open(html_file, 'r') as file:
        html_content = file.read()

    start_marker = '<pre id="map-sketch">'
    end_marker = '</pre>'

    start_index = html_content.find(start_marker) + len(start_marker)
    end_index = html_content.find(end_marker, start_index)

    if start_index == -1 or end_index == -1:
        raise ValueError("Markers for map sketch not found in HTML file.")

    new_html_content = html_content[:start_index] + '\n' + \
        map_sketch_content + '\n' + html_content[end_index:]

    with open(html_file, 'w') as file:
        file.write(new_html_content)


def monitor_file_changes(sketch_file, html_file):
    # Get the initial modification time of the files
    last_modified_sketch = os.path.getmtime(sketch_file)

    while True:
        # Check if the sketch file has been modified
        current_modified_sketch = os.path.getmtime(sketch_file)
        if current_modified_sketch != last_modified_sketch:
            print("Sketch file modified. Updating HTML and running command...")
            update_html_with_map_sketch(html_file, sketch_file)
            run_command()
            print("Command executed successfully.")
            last_modified_sketch = current_modified_sketch

        # Sleep for a short duration before checking again
        time.sleep(1)


if __name__ == "__main__":
    sketch_file = "sketch.map"
    html_file = "map_painter.html"
    monitor_file_changes(sketch_file, html_file)
