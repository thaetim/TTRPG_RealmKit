import subprocess
import time
import yaml
from pathlib import Path


def load_command_config():
    command_file = Path('command_config.yaml')
    if not command_file.exists():
        raise FileNotFoundError(f"{command_file} not found.")

    with open(command_file, 'r') as file:
        config = yaml.safe_load(file)

    return config


def construct_command(config, sketch_file, output_file):
    executable = config['executable']
    arguments = config['arguments']
    input_redirection = config.get('input_redirection', '')

    # Replace placeholders in arguments
    formatted_arguments = []
    for arg in arguments:
        for key, value in arg.items():
            formatted_arguments.append(
                f"{key} {value.format(sketch_file=sketch_file, output_file=output_file)}")

    # Construct the command
    command = f"{executable} " + " ".join(formatted_arguments) + \
        f" {input_redirection.format(sketch_file=sketch_file)}"
    return command


def run_command(show_output=True):
    # Define the paths
    planet_dir = Path('planet')
    root_dir = Path(__file__).parent.resolve()  # Get the root directory
    sketch_file = root_dir / 'sketch.map'
    output_file = root_dir / 'preview.bmp'

    # Load command configuration
    config = load_command_config()

    # Construct the command to run
    command = construct_command(config, sketch_file, output_file)

    # Determine the output settings
    stdout_setting = None if show_output else subprocess.DEVNULL
    stderr_setting = None if show_output else subprocess.DEVNULL

    # Ensure the working directory exists
    if not planet_dir.exists():
        raise FileNotFoundError(f"{planet_dir} directory not found.")

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
    last_modified_sketch = Path(sketch_file).stat().st_mtime

    while True:
        # Check if the sketch file has been modified
        current_modified_sketch = Path(sketch_file).stat().st_mtime
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
