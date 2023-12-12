# py_vb_toolbox_update
Vogt-Bailey index toolbox in Python

# Data collection:
You can find a collection of the required files to run the vb_index.py in the following Google Drive link:
```bash
https://drive.google.com/drive/folders/18auvkm7pFqf87dIUnHxSk_cB_HWPUYr_
```

# Commands:

## Full brain analysis:
```bash
./vb_index.py --surface R.midthickness.k.fix.surf.gii --mask R.cortical.vertices.k.fix.shape.gii --data R.NOISE_cubic_low_res_k.fix.func.gii --output full_brain_test --full-brain
```

## Searchlight analysis:
```bash
./vb_index.py --surface R.midthickness.k.fix.surf.gii --mask R.cortical.vertices.k.fix.shape.gii --data R.NOISE_cubic_low_res_k.fix.func.gii --output searchlight_test
```
