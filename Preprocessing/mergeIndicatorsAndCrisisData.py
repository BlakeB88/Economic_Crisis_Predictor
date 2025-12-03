# Step 1: Import pandas
import pandas as pd

INDICATORS_FILE = 'all_indicators_merged.csv' 

CRISIS_FILE = '20160923_global_crisis_data.xlsx' 

CRISIS_COLUMN_NAME = (
    "SOVEREIGN EXTERNAL DEBT 2: DEFAULT and RESTRUCTURINGS, 1800-2012--"
    "Does not include defaults on WWI debt to United States and United Kingdom "
    "but includes post-1975 defaults on Official External Creditors"
)

NEW_TARGET_COLUMN_NAME = 'External_Debt_Crisis'

try:
    indicators_df = pd.read_csv(INDICATORS_FILE)
    print(f"-> Success. Shape of indicators data: {indicators_df.shape}")
except FileNotFoundError:
    print(f"!!! ERROR: File not found: {INDICATORS_FILE}. Please check the filename.")
    exit()
try:
    crisis_df = pd.read_excel(
        CRISIS_FILE,
        usecols=['Country', 'Year', CRISIS_COLUMN_NAME]
    )
    print(f"-> Success. Shape of raw crisis data: {crisis_df.shape}")
except FileNotFoundError:
    print(f"!!! ERROR: File not found: {CRISIS_FILE}. Please check the filename.")
    exit()
except ValueError as e:
    print(f"!!! ERROR: A column was not found. Check your column names. Details: {e}")
    exit()
crisis_df = crisis_df.rename(columns={
    CRISIS_COLUMN_NAME: NEW_TARGET_COLUMN_NAME
})
crisis_df[NEW_TARGET_COLUMN_NAME] = crisis_df[NEW_TARGET_COLUMN_NAME].fillna(0)
crisis_df[NEW_TARGET_COLUMN_NAME] = crisis_df[NEW_TARGET_COLUMN_NAME].astype(int)
crisis_df['Year'] = pd.to_numeric(crisis_df['Year'], errors='coerce')

final_merged_df = pd.merge(
    indicators_df,
    crisis_df,
    on=['Country', 'Year'],
    how='inner'
)
final_merged_df = final_merged_df.sort_values(by=['Country', 'Year'])
countries_in_final_set = final_merged_df['Country'].unique()
output_filename = 'indicator_merged_dataset.csv'
final_merged_df.to_csv(output_filename, index=False)
