import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np

INPUT_FILE = 'indicator_merged_dataset.csv'
OUTPUT_FILE = 'imputed_dataset.csv'

indicator_columns = [
    'gdp_growth', 
    'inflation', 
    'domestic_credit', 
    'current_account', 
    'total_reserves'
]

id_columns = ['Country', 'Year', 'External_Debt_Crisis']

df = pd.read_csv(INPUT_FILE)

df = df.sort_values(by=['Country', 'Year'])

imputed_df_list = []
all_countries = df['Country'].unique()
for i, country in enumerate(all_countries):
    country_df = df[df['Country'] == country].copy()
    data_to_impute = country_df[indicator_columns]
    if data_to_impute.isnull().values.any():
        imputer = IterativeImputer(
            max_iter=10,            
            random_state=0,         
            skip_complete=True     
        )
        imputed_data = imputer.fit_transform(data_to_impute)
        country_df[indicator_columns] = imputed_data
    imputed_df_list.append(country_df)

final_imputed_df = pd.concat(imputed_df_list)
final_imputed_df = final_imputed_df.sort_values(by=['Country', 'Year'])

final_imputed_df.info()
final_imputed_df.to_csv(OUTPUT_FILE, index=False)
