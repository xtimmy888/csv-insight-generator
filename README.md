# CSV Insight Generator

A fast, lightweight Streamlit app that profiles any CSV file and gives instant insights:
- dataset overview (rows/cols, memory usage, duplicates)
- schema + dtypes
- missing values table
- numeric summary stats
- categorical top values
- correlations + top correlated pairs
- basic plots (histograms, bar charts, correlation heatmap)
- download the cleaned CSV (after optional datetime parsing)

## Demo
**What you do:**
1. Upload a CSV
2. Adjust separator/encoding if needed
3. Explore preview + stats + plots
4. Download the cleaned CSV

> Add screenshots here after you run it:
- `screenshots/overview.png`
- `screenshots/correlations.png`

## Requirements
- Python 3.10+ recommended
- Dependencies listed in `requirements.txt`

## Setup (Windows)
### 1) Create and activate a virtual environment
**PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate