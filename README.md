# Digital Image Processing

This repository contains two image-processing tasks implemented with Python,
OpenCV, NumPy, and Matplotlib.

## Tasks

- `task1代码.py`: vessel-like structure enhancement for grayscale images. It estimates a
  Gaussian point-spread function from a crosshair marker, applies Wiener and
  TV-regularized Richardson-Lucy restoration, then enhances vessel detail with
  Frangi, Gabor, top-hat, and guided-filter stages.
- `task2代码.py`: particle counting for color images. It uses thresholding,
  morphology, watershed splitting, and a circle-distance rule to report total
  particles and non-overlapping particles.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place input images in `data/`. The original assignment images are not committed.

Expected default paths:

```text
data/FigP0520.tif
data/222.jpg
```

## Usage

Run vessel enhancement:

```bash
python "task1代码.py" --input data/FigP0520.tif --output-dir outputs
```

Run particle counting:

```bash
python "task2代码.py" --input data/222.jpg --output-dir outputs
```

Use `--show` to display diagnostic plots. Generated images are written to
`outputs/`.

## Tests

```bash
python -m unittest discover -s tests
```

The tests cover import safety and small utility behavior. They do not require
the original assignment images.

