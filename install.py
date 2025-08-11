import sys
from pathlib import Path


def install():
    module_name = "materialite"
    module_dir = "source"

    # Get the absolute path to the root directory of the repository
    root_path = Path(".").resolve()

    # Look through the user's path for a site-packages location
    for path in sys.path:
        if path.endswith("site-packages"):
            package_path = Path(path)
            print(path)
            break

    # Create path for <module name>.pth
    module_pth_path = package_path / f"{module_name}.pth"

    # Get absolute path of parent directory
    module_dir_path = Path(module_dir).resolve()

    # Create an export line to add to the bashrc
    module_export_line = "export PYTHONPATH=$PYTHONPATH:" + str(module_dir_path)

    if module_pth_path.exists() or is_line_in_bashrc(module_export_line):
        print(str(module_dir_path) + " is already installed")
    else:
        try:
            # Add the parent module directory to the pth file
            with open(module_pth_path, "w") as file_module_pth:
                file_module_pth.write(str(module_dir_path))
                print(f"Added {module_dir_path} to the Python system path")

        except OSError:
            # Add to the PYTHONPATH in bashrc if cannot go the Anaconda site-packages route
            with open(Path.home() / ".bashrc", "a") as bashrc_file:
                bashrc_file.write(
                    f"\n# Added {module_name} module path through install.py"
                )
                bashrc_file.write("\n" + module_export_line + "\n")
                print(f"Added {module_dir_path} to the PYTHONPATH through ~/.bashrc")


def is_line_in_bashrc(line):
    bashrc_path = Path.home() / ".bashrc"
    if bashrc_path.exists():
        with open(bashrc_path) as bashrc_file:
            return True if line in bashrc_file.read() else False


install()
