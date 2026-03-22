# MeshNet: A Mesh-based Graph Neural Network Framework for Structural Analysis

## MeshNet vs FEM
### Hardware
- CPU: Intel Core i7-12700
- GPU: Nvidia RTX 5060

Evaluation of 250 graphs
- Cuboid1: 528 nodes (coarse), 21480 nodes (fine)
  - Graph construction: 1.58s (coarse)
  - MeshNet: 1.45s (coarse)
  - FEM: 9.96s (coarse), 479.44s (fine)
- Cuboid3: 1193 nodes (coarse), 54779 nodes (fine)
  - Graph construction: 4.2s (coarse)
  - MeshNet: 4.03s (coarse)
  - FEM: 19.95s (coarse)



## How to Use

Create environment:
```
conda env create -f environment.yaml
conda activate meshnet
pip install -r requirements.txt
```

Generate volume meshes with two different resolutions:
```bash
python meshgen.py step --input meshes/primitives/step --output meshes/primitives/msh --size 0.005  --element-order 1
python meshgen.py step --input meshes/primitives/step --output meshes/primitives/msh --size 0.0025  --element-order 2
```

Generate datasets:
```bash
python data.py meshes/primitives/msh --num_samples 100
python data.py meshes/factory/msh/HexNut2_cg1.msh --num_samples 100 --num_contacts 2
```

Train:
```bash
python train.py --dataset Cuboid200 --epochs 50 --learning-rate 1e-4 --batch-size 64 --tensorboard --layers 10
python train.py --dataset Cuboid \
 --epochs 500 \
 --learning-rate 1e-4 \
 --batch-size 64 \
 --tensorboard \
 --weighted-loss \
 --alpha 20 \
 --target stress \
```

Play:
```bash
python play.py --checkpoint Mix250_all_w --dataset Test100-2/Bushing3_100 --plots -n 5
```