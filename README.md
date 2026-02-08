```
python meshgen.py step --input meshes/dataset/step --output meshes/dataset/msh --size 0.005  --element-order 1
python meshgen.py step --input meshes/dataset/step --output meshes/dataset/msh --size 0.002  --element-order 2
python data.py meshes/dataset/msh --num_samples 100
python train.py --dataset dataset --num-epochs 500 --learning-rate 1e-4 --batch-size 64 --tensorboard
python play.py --checkpoint Cuboid1_100_stress_uw --dataset Cuboid1_100 --plots -n 5 --mesh-dir meshes/dataset
python train.py --dataset dataset --num-epochs 500 --learning-rate 1e-4 --batch-size 64 --tensorboard --target stress --weighted-loss --alpha 20
```