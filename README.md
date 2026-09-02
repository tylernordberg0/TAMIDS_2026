# Predicting NFL Draft Success and Identifying Undervalued Offensive Prospects

A STAT 421 (TAMIDS 2026) project that uses NFL Combine measurables and college
production statistics to (1) predict whether a drafted offensive player
(QB, RB, or WR) will "hit" or "bust" in the NFL, (2) predict where a player
should be drafted based on his measurables/production, and (3) combine both
signals to flag **undervalued prospects** — players who outperformed their
draft slot relative to what the model expected.

## Approach

For each position group we train two models:

1. **Hit/bust classifier** — predicts the probability a player has a
   successful NFL career (defined by a position-specific weighted approximate
   value, `w_av`, threshold), using combine measurables and college stats.
   Random Forest, Gradient Boosting, and Logistic Regression are compared,
   with prediction thresholds tuned per position to balance hit precision vs.
   recall.
2. **Draft position regressor** — predicts expected draft pick using the same
   features (Lasso/Ridge regression), reported primarily via cross-validated
   MAE/RMSE/R².

The two models are then combined into a **value score** (hit probability +
draft pick difference) to surface players who were both likely to succeed
and drafted later than expected — e.g. the QB model flags Tom Brady
(pick 199) and Dak Prescott (pick 135); the RB model flags Aaron Jones
(pick 182) and Michael Turner (pick 154).

## Repository Structure

```
models/
  QB_model.ipynb        Quarterback hit/bust + draft value model
  RB_model.ipynb        Running back hit/bust + draft value model
  WR_model.ipynb        Wide receiver hit/bust + draft value model
college_stats/
  passing/               Season-by-season college passing stats
  receiving/             Season-by-season college receiving stats
  rushing/                Season-by-season college rushing stats
pdfs for report/         Exported visualizations used in the write-up
421_final_paper/          Final STAT 421 paper (LaTeX + PDF)
nfl_draft_combine.csv     Combine measurables + draft/outcome data
merged_data.csv           Combine + college stats merged dataset
merged_data_with_college.csv  Extended merged dataset
school-conferences.txt    School-to-conference lookup used for feature engineering
TAMIDS_2026.ipynb          Exploratory data analysis notebook
TAMIDS_2026.pdf             Project report/summary
```

## Data

- **Combine/draft data**: `nfl_draft_combine.csv` — combine measurables
  (height, weight, 40-yard dash, bench, vertical, broad jump, cone, shuttle),
  draft round/pick, and career outcome metrics (Pro Bowls, All-Pro, weighted
  approximate value, etc.).
- **College production data**: `college_stats/{passing,receiving,rushing}/`
  — season-level college statistics scraped/compiled per year.

## Getting Started

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

Open `TAMIDS_2026.ipynb` for the exploratory analysis, or any notebook under
`models/` for the position-specific hit/bust and draft-value models.

## Report

The final write-up is available in `421_final_paper/` (LaTeX source and
compiled PDF) and as `TAMIDS_2026.pdf` at the repository root, with
supporting figures in `pdfs for report/`.
