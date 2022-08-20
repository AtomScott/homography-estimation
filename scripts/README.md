# Console Scripts

All files ending in ``.py`` in the ``scripts`` directory are meant to be executable scripts. setup.py contains a short scripts that automatically adds each script as an entry point to the package.

Make sure that each file contains a `def main()` function that is executed when the script is run.

```python
def setup_console_scripts():
    script_dir = Path(__file__).parent / "scripts"
    console_scripts = []
    for pyfile in script_dir.glob("*.py"):
        script_name = pyfile.stem
        console_scripts.append(f"{script_name}=scripts.{script_name}:main")
    return console_scripts
```
