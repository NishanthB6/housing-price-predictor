# Housing Price Predictor

Predicting `SalePrice` for the Ames, Iowa housing dataset (79 features, regression task).

## Project structure

```
housing-price-predictor/
├── data/               # train.csv, test.csv, sample_submission.csv, data_description.txt
├── notebooks/          # EDA and experimentation notebooks
├── src/                # reusable python modules (cleaning, features, modeling)
├── outputs/            # generated submission.csv files, saved models
└── requirements.txt
```

## Setup in VS Code

1. **Open the folder** in VS Code: `code housing-price-predictor`

2. **Create a virtual environment** (from the integrated terminal, in the project root):
   ```bash
   python -m venv .venv
   ```

3. **Activate it**:
   - macOS/Linux: `source .venv/bin/activate`
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`

4. **Select the interpreter in VS Code**: Cmd/Ctrl+Shift+P → "Python: Select Interpreter" → choose the one inside `.venv`.

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

6. **Install the VS Code extensions** (if not already installed): Python (Microsoft), Jupyter.

7. **Run the starter script** to sanity-check everything loads:
   ```bash
   python src/eda_starter.py
   ```
   Or open `notebooks/` in VS Code and start a Jupyter notebook there for interactive EDA — either works with this setup.

## Workflow from here

1. **EDA** (`notebooks/`) — explore distributions, missingness, correlations with `SalePrice`.
2. **Cleaning / feature engineering** (`src/`) — handle the "NA means none" columns, encode ordinal quality fields, engineer any new features.
3. **Modeling** (`src/`) — start with a baseline (e.g. linear regression or random forest), then iterate.
4. **Submission** (`outputs/`) — predict on `test.csv`, format to match `sample_submission.csv`, save as `outputs/submission.csv`.
