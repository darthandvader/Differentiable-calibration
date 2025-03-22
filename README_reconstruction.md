# Install

```
cd submodules/hunyuan3d
pip install -r requirements.txt
pip install -e .
# for texture
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
```

# Inference

```
python scripts/run_hunyuan3d.py
```