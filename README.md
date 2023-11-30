# py_vb_toolbox_update
Vogt-Bailey index toolbox in Python

# Commands:

## Full brain analysis:
```bash
./vb_index.py --surface R.midthickness.k.fix.surf.gii --mask R.cortical.vertices.k.fix.shape.gii --data R.NOISE_cubic_low_res_k.fix.func.gii --output full_brain_test --full-brain
```

## Searchlight analysis:
```bash
./vb_index.py --surface R.midthickness.k.fix.surf.gii --mask R.cortical.vertices.k.fix.shape.gii --data R.NOISE_cubic_low_res_k.fix.func.gii --output searchlight_test
```
