# Semi-supervised Meta-learning for Multi-source Heterogeneity in Time-series Data

**Original Repository**: [lidazhang/ssml-time-series-heterogeneity](https://github.com/lidazhang/ssml-time-series-heterogeneity)  
**Published at**: MLHC 2023  
**Paper**: [View PDF](https://static1.squarespace.com/static/59d5ac1780bd5ef9c396eda6/t/64d1abf2bbaa8263069c3a2c/1691462642799/ID153_Research+Paper_2023.pdf)

---

## Training Instructions

### Logistic Regression Models

```bash
# In-hospital mortality prediction
python -um mimic3models.in_hospital_mortality.logistic.main --l2 --C 0.001 --output_dir mimic3models/in_hospital_mortality/logistic

# Decompensation prediction
python -um mimic3models.decompensation.logistic.main --output_dir mimic3models/decompensation/logistic

# Length of stay prediction
python -um mimic3models.length_of_stay.logistic.main_cf --output_dir mimic3models/length_of_stay/logistic

```

### LSTM
```bash
# In-hospital mortality prediction
python -um mimic3models.in_hospital_mortality.main \
  --network mimic3models/keras_models/lstm.py \
  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 \
  --mode train --batch_size 8 \
  --output_dir mimic3models/in_hospital_mortality

# Decompensation prediction
python -um mimic3models.decompensation.main \
  --network mimic3models/keras_models/lstm.py \
  --dim 128 --timestep 1.0 --depth 1 \
  --mode train --batch_size 8 \
  --output_dir mimic3models/decompensation

# Length of stay prediction
python -um mimic3models.length_of_stay.main \
  --network mimic3models/keras_models/lstm.py \
  --dim 64 --timestep 1.0 --depth 1 --dropout 0.3 \
  --mode train --batch_size 8 --partition custom \
  --output_dir mimic3models/length_of_stay
