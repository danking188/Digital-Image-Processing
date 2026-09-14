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

Use Python **3.11 or 3.12**, and run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
```

Place input images in `data/`. The original assignment images are not committed.
On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.
`requirements.txt` declares dependency ranges; `requirements.lock` pins versions for reproducible installation.

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

Count only a region of interest (x, y, width, height):

```bash
python "task2代码.py" --input data/222.jpg --roi 0 0 200 200 --output-dir outputs/roi
```

The ROI must lie fully inside the image and be at least 2 × 2 pixels. Invalid input and failed output writes raise errors instead of reporting success.

## Outputs and assumptions

| Pipeline | Output files |
| --- | --- |
| Vessel enhancement | `vessel_enhanced.png`, `vessel_restored_tvrl.png`, `estimated_psf.png` |
| Particle counting | `particles_all.png`, `particles_nonoverlap.png`, `particles_binary.png`, `particles_watershed_boundary.png` |

- Repeated runs overwrite these filenames; use separate `--output-dir` values for separate inputs.
- Vessel enhancement assumes a calibration crosshair in the bottom-right region and bright vessel-like structures. Inputs must be at least 70 × 70 pixels for template matching; arbitrary images without a crosshair are not a validated use case.
- Particle counting assumes bright particles on a darker background. Touching, irregular, low-contrast or dark objects may require parameter tuning; the circle-distance rule is an approximation, not ground truth.
- Both tasks run without opening plot windows by default; use `--show` only in a desktop graphical session. These are educational algorithms, not clinical imaging software.

## Tests

```bash
python -m unittest discover -s tests
```

The 12 tests cover utilities, invalid ROIs/PSFs, empty backgrounds, failed output writes, a no-shift Wiener identity regression, and both complete pipelines using generated synthetic images. Generated outputs are decoded again to verify they are readable. Tests require no original assignment images.

GitHub Actions runs the same tests and both CLI help commands on Python 3.11/3.12. Synthetic tests verify execution and specific behaviors; visual quality and counting accuracy on the original assignment images still require those images and reference labels.
