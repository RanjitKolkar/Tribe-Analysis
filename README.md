# Tribe Data Analysis (Streamlit)

This project analyzes two tribe datasets:
- `Nirankal Tribe List.xlsx`
- `Virnoda Tribe List.xlsx`

The Streamlit app compares:
- Population and sex distribution
- Age distribution
- Anthropometric measurements (BMI, height, weight, etc.)
- Questionnaire responses (education, work, tobacco/alcohol use)
- Data completeness by sheet

## Quick Data Understanding

Based on the current files:
- Nirankal roster size: `120`
- Virnoda roster size: `68`
- Both tribes include male and female records
- Median age is close (`17` vs `18`)
- Median BMI differs (`15.95` in Nirankal vs `17.6` in Virnoda)
- Tobacco non-"never" responses are higher in Virnoda (`58.8%`) than Nirankal (`33.3%`)
- Alcohol non-"never" responses are minimal in Nirankal (`0%`) and higher in Virnoda (`19.6%`)

## Run the App

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Start Streamlit:

```powershell
streamlit run app.py
```

## Notes

- The app auto-detects all `.xlsx` files in this folder and treats each workbook as one tribe.
- It handles minor sheet naming differences (e.g., `Questionnaire` vs `Questionnaire `).
- Some workbook sheets contain sparse fields; the app includes a data quality tab to highlight completeness.
