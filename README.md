# Tomato Leaf Disease Detection and Classification (Notebook-Only)

This project is intentionally minimal and centered around one final notebook.

## What This Project Contains
- `notebooks/02_tomato_clean_pipeline.ipynb`
  - Full pipeline: preprocessing, segmentation, feature extraction, ML, DL, evaluation, plots, inference.
- `prepared_tomato_data/`
  - Local dataset split into `train/`, `val/`, `test/`.
- `requirements.txt`
  - Python dependencies used by the notebook.

## Required Files to Run on Another Environment
Keep/copy exactly these:
1. `notebooks/02_tomato_clean_pipeline.ipynb`
2. `requirements.txt`
3. `prepared_tomato_data/` (with all images and class folders)

## Expected Dataset Structure
The notebook expects this exact layout:

```text
prepared_tomato_data/
  train/
    Tomato___healthy/
    Tomato___Early_blight/
    Tomato___Late_blight/
    Tomato___Leaf_Mold/
  val/
    Tomato___healthy/
    Tomato___Early_blight/
    Tomato___Late_blight/
    Tomato___Leaf_Mold/
  test/
    Tomato___healthy/
    Tomato___Early_blight/
    Tomato___Late_blight/
    Tomato___Leaf_Mold/
```

## Python Version (Important)
Use **Python 3.12** (recommended).

Why: this project uses `numpy<2`; with Python 3.14, pip may fail while trying to build NumPy from source.

## Setup and Run (Windows PowerShell)
```powershell
cd <your-project-folder>
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install notebook
jupyter notebook notebooks/02_tomato_clean_pipeline.ipynb
```

Then open the notebook and click **Run All**.

## Setup and Run (Linux/macOS)
```bash
cd <your-project-folder>
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install notebook
jupyter notebook notebooks/02_tomato_clean_pipeline.ipynb
```

Then run all cells from top to bottom.

## Non-Interactive Execution (Optional)
Useful for validation in CI or headless servers:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_tomato_clean_pipeline.ipynb
```

## What Gets Generated
After execution, the notebook creates:
- `artifacts_notebook/ml_metrics.json`
- `artifacts_notebook/dl_metrics.json`
- `artifacts_notebook/comparison.csv`
- `artifacts_notebook/best_dl_model.pt`

These are generated files and can be deleted/recreated anytime by rerunning the notebook.

## Common Issues
1. `No module named ...`
   - Your virtual environment is not active or dependencies were not installed.
2. `Dataset folders are missing`
   - Check `prepared_tomato_data/train`, `val`, `test` paths and class folder names.
3. NumPy install/build error on Windows
   - Use Python 3.12 as shown above.
