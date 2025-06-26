# Roi Analysis
Simple library for analysing DAPI microscopy images.

## Prerequisites
Python 3.12 and uv.

## Installation
You can install the library from github:

```bash
pip install git+https://github.com/kuko6/roi-analyser.git@library
```

or build it locally:

```bash
git clone https://github.com/kuko6/roi-analyser.git@library
cd roi-analyser
pip install -e .
```

## Development
To set up the development environment:

1. Clone the repository:
```bash
git clone https://github.com/kuko6/roi-analyser.git@library
cd roi-analyser
```

2. Create virtual environment:
```bash
uv sync
```

3. Install the dev dependencies:
```bash
uv sync --dev
```

4. Run with:
```bash
uv run roi_analyser/processing.py
```
