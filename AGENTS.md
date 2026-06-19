# Repository Guidelines

## Project Structure & Module Organization
This repository is a small script-based Python project for a neural-network car simulation built with Pygame. `nnCarGame.py` is the main runtime: it opens the game window, evolves cars, and loads generated tracks. `mapGen.py` is a standalone track generator that writes `randomGeneratedTrackFront.png` and `randomGeneratedTrackBack.png`. Static assets live under `Images/`, with car sprites in `Images/Sprites/` and track tiles in `Images/TracksMapGen/`. Root-level `.png` files are runtime backgrounds or generated outputs, not source code.

## Build, Test, and Development Commands
There is no packaged build system in this repo. Use a local virtual environment and run the scripts directly:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pygame numpy pillow shapely
python3 nnCarGame.py
python3 mapGen.py
```

`nnCarGame.py` launches the simulation. `mapGen.py` regenerates the two track overlay images used by the main game.

## Coding Style & Naming Conventions
Follow the existing codebase style when editing: Python, mostly module-level scripts, 4-space indentation for new code, and concise inline comments only where logic is not obvious. Preserve existing public names unless you are refactoring broadly. Use `snake_case` for new functions and variables, `CamelCase` for classes, and keep asset paths relative to the repository root, for example `Images/Sprites/white_small.png`.

## Testing Guidelines
This repository currently has no automated test suite. For changes to simulation logic, verify by running `python3 nnCarGame.py` and exercising the keyboard controls in the window. For changes to map generation, run `python3 mapGen.py` and confirm both generated track images update correctly. If you add tests, prefer `pytest` with files named `test_<module>.py`.

## Commit & Pull Request Guidelines
The current Git history uses short summaries such as `initial commit` and `third commit`. Improve on that pattern: write brief, imperative commit subjects like `Fix collision reset on generated tracks`. Pull requests should include a clear behavior summary, manual test steps, and screenshots or short recordings for visible gameplay or rendering changes.

## Asset & Configuration Notes
Do not rename or move referenced image assets without updating the hard-coded load paths in both Python scripts. Keep generated track images in the repository root unless you also update the loader logic in `nnCarGame.py`.
