import subprocess
import threading
import yaml
from pathlib import Path


def load_yaml(file_path):
    """Load YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def construct_command(base_config, alterations, sketch_file, output_file):
    """Construct command string with alterations."""
    executable = base_config['executable']
    arguments = base_config['arguments']
    input_redirection = base_config.get('input_redirection', '')

    # Replace placeholders in arguments
    formatted_arguments = []
    for arg in arguments:
        for key, value in arg.items():
            formatted_value = value.format(
                sketch_file=sketch_file, output_file=output_file)
            # Handle flags (arguments without values) separately
            if value == "":
                formatted_arguments.append(f"{key}")
            else:
                formatted_arguments.append(f"{key} {formatted_value}")

    # Apply alterations
    # Use a set to track which arguments have been replaced
    existing_args = set(arg.split()[0] for arg in formatted_arguments)
    for alteration in alterations:
        for key, value in alteration.items():
            if value == "":
                # Add the flag if it's not already present
                if key not in existing_args:
                    formatted_arguments = [f"{key}"] + formatted_arguments
            else:
                # Replace the existing argument or add a new one
                formatted_arguments = [
                    f"{key} {value}" if arg.startswith(key) else arg
                    for arg in formatted_arguments
                ]
                if key not in existing_args:
                    formatted_arguments.append(f"{key} {value}")

    # Construct the command
    command = f"{executable} " + " ".join(formatted_arguments) + \
        f" {input_redirection.format(sketch_file=sketch_file)}"
    return command


def run_command(command):
    """Run a command."""
    print(f"Running command: {command}")
    subprocess.run(command, shell=True, cwd=planet_dir,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def execute_layers(base_config, resolution, versions, sketch_file):
    """Execute commands for each version in separate threads."""
    threads = []
    for version in versions:
        # Apply resolution settings to alterations
        alterations = version.get('alterations', [])
        for res in resolution:
            alterations.append(res)

        version_name = version.get('name', 'default')
        output_file = root_dir / 'renders' / f'{version_name}.bmp'
        command = construct_command(
            base_config, alterations, sketch_file, output_file)
        thread = threading.Thread(target=run_command, args=(command,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    # Define paths
    planet_dir = Path('planet')
    root_dir = Path(__file__).parent.resolve()
    sketch_file = root_dir / 'sketch.map'

    # Load configurations
    base_config = load_yaml('command_config.yaml')
    config = load_yaml('command_alterations.yaml')
    resolution = config.get('resolution', [])
    versions = config.get('versions', [])

    # Execute layers
    execute_layers(base_config, resolution, versions, sketch_file)
