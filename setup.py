from setuptools import find_packages, setup
from pathlib import Path


def setup_console_scripts():
    script_dir = Path(__file__).parent / "scripts"
    console_scripts = []
    for pyfile in script_dir.glob("*.py"):
        script_name = pyfile.stem
        console_scripts.append(f"{script_name}=scripts.{script_name}:main")
    return console_scripts


with open("requirements.txt") as requirements_file:
    install_requirements = requirements_file.read().splitlines()


setup(
    name="dlhe",
    version="0.0.1",
    packages=find_packages(),
    entry_points={"console_scripts": setup_console_scripts()},
)
