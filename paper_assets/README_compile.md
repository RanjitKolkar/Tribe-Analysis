# Nature-Style Manuscript Build Notes

Main file:
- `nature_manuscript.tex`

Bibliography:
- `references.bib`

Figures used in manuscript:
- `figures/methodology_pipeline.png`
- `figures/age_sex_structure.png`
- `figures/effect_sizes_cohens_d.png`
- `figures/bmi_violin.png`
- `figures/behavior_non_never_rates.png`
- `figures/fingerprint_top_codes.png`
- `figures/fingerprint_symmetry.png`
- `figures/anthropometry_pca.png`

Tables generated from data:
- `tables/table1_cohort_profile.csv`
- `tables/table2_anthropometry_stats.csv`
- `tables/table3_categorical_stats.csv`
- `tables/table4_fingerprint_profile.csv`
- `tables/table5_fingerprint_symmetry.csv`
- `tables/table6_missingness.csv`

## Compile Commands

If `sn-jnl.cls` is available in your TeX installation or local folder:

```powershell
cd paper_assets
pdflatex -interaction=nonstopmode nature_manuscript.tex
bibtex nature_manuscript
pdflatex -interaction=nonstopmode nature_manuscript.tex
pdflatex -interaction=nonstopmode nature_manuscript.tex
```

## If Build Fails with `sn-jnl.cls` Missing

The manuscript is already formatted for Springer Nature (`sn-jnl`).
Install/copy `sn-jnl.cls` (and associated template files) into your TeX path or this folder, then rerun the commands above.
