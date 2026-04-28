Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

python train_tomato_models.py `
  --skip-scratch `
  --batch-size 64 `
  --image-size 128 `
  --head-epochs 5 `
  --finetune-epochs 5 `
  --materialize-root prepared_tomato_data `
  --output-dir artifacts_final
