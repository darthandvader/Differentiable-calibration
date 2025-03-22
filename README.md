# Conda environment setup

```
conda create --name diffcali python=3.10
conda activate diffcali
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
Run `pip install -e .` in the root directory of this project to install `diffcali` as a library.
# Calibration with differentiable rendering

## Quick start:
Download test data from [Google drive](https://drive.google.com/file/d/1DFUI_d4ouyvCbLwPtTMGt5AVkI4t26gf/view?usp=drive_link), and put it under `/data`

Run:
```python
python scripts/origin_retracing.py
```

## Generate single-view reconstruction

Follow [README_reconstruction.md](README_reconstruction.md)