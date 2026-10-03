# Data

`raw/ai4i2020.csv` is the **AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning
Repository, dataset id 601), redistributed unchanged under its CC BY 4.0 licence.

- Source: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- Citation: Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance
  Applications*. Third International Conference on Artificial Intelligence for Industries
  (AI4I 2020). https://doi.org/10.1109/AI4I49448.2020.00023
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
- sha256 of the CSV: `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`
  (pinned in `configs/pipeline.yaml`; the ingest stage refuses any other file)

The dataset is **synthetic**: it was generated to reflect real milling-machine failure
behaviour, and it contains no personal data. 10,000 rows, 14 columns, 339 failures (3.39 %).

To re-download and verify instead of using the committed copy:

```bash
rm data/raw/ai4i2020.csv
pm-pipeline validate      # ingest downloads the UCI zip, extracts the CSV and checks the sha256
```
