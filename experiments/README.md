# Experiments

## atlas_experiment/

Real ATLAS particle physics data. Download and extract with `download.py`.

```bash
cd atlas_experiment
python download.py --all-steps  # downloads 2.8GB, creates atlas_200m.bin
```

Then train:
```bash
python train_example.py --data experiments/atlas_experiment/atlas_200m.bin --epochs 10
```
