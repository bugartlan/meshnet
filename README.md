Create environment:
```
conda env create -f environment.yaml
conda activate meshnet
pip install -r requirements.txt
```

Generate volume meshes:
```bash
python meshgen.py step --input meshes/Custom/step --output meshes/Custom/msh --size 0.005  --element-order 1
python meshgen.py step --input meshes/Custom/step --output meshes/Custom/msh --size 0.0025  --element-order 2
```

Generate datasets:
```bash
python data.py meshes/Custom/msh --num_samples 100
python data.py meshes/Custom/msh/Cuboid3_cg1.msh --num_samples 100
```

Train:
```bash
python train.py --dataset Cuboid200 --num-epochs 50 --learning-rate 1e-4 --batch-size 64 --tensorboard
python train.py --dataset Cuboid \
 --num-epochs 500 \
 --learning-rate 1e-4 \
 --batch-size 64 \
 --tensorboard \
 --weighted-loss \
 --alpha 20 \
 --target stress \
```

Play:
```bash
python play.py --checkpoint Cuboid_all_uw --dataset Cuboid1_100 --plots -n 5
```