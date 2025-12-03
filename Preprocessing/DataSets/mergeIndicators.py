import pandas as pd
import os

indicators_to_load = {
    'GDPAnnualGrowth.csv': 'gdp_growth',
    'Inflation.csv': 'inflation',
    'FiscalBalance.csv': 'fiscal_balance',
    'LendingInterestRate.csv': 'lending_rate',
    'DomesticCreditToPrivateSector.csv': 'domestic_credit',
    'GovernmentGrossDebt.csv': 'govt_debt',
    'CurrentAccountBalance.csv': 'current_account',
    'TotalReserves.csv': 'total_reserves'
}

def load_and_clean_indicator(file_name, new_col_name):
    """
    Loads a World Bank CSV, selects the 3 key columns,
    and renames OBS_VALUE to a unique name.
    """
    print(f"Processing {file_name}...")
    try:
        # Load the CSV file
        df = pd.read_csv(file_name)
        
        # 1. Select only the columns we care about
        df_cleaned = df[['REF_AREA_LABEL', 'TIME_PERIOD', 'OBS_VALUE']]
        
        # 2. Rename OBS_VALUE to our unique name (e.g., 'gdp_growth')
        df_renamed = df_cleaned.rename(columns={
            'OBS_VALUE': new_col_name,
            'REF_AREA_LABEL': 'Country', # Standardize common columns
            'TIME_PERIOD': 'Year'       # Standardize common columns
        })
        
        # 3. Convert Year to integer for clean merging
        df_renamed['Year'] = pd.to_numeric(df_renamed['Year'], errors='coerce')
        
        print(f"-> Finished {file_name}. Found {len(df_renamed)} data points.")
        return df_renamed
        
    except FileNotFoundError:
        print(f"!!! ERROR: File not found: {file_name}. Please check the filename.")
        return None
    except KeyError as e:
        print(f"!!! ERROR: Column not found in {file_name}. Missing {e}. Check schema.")
        return None

file_list = list(indicators_to_load.keys())
first_file_name = file_list[0]
first_col_name = indicators_to_load[first_file_name]
merged_df = load_and_clean_indicator(first_file_name, first_col_name)

if merged_df is not None:
    for file_name in file_list[1:]:
        col_name = indicators_to_load[file_name]
        next_df = load_and_clean_indicator(file_name, col_name)
        
        if next_df is not None:
            merged_df = pd.merge(
                merged_df, 
                next_df, 
                on=['Country', 'Year'], 
                how='outer'
            )
            print(f"-> Merged {file_name}. Shape is now: {merged_df.shape}")

    print("\n--- Merge Complete! ---")
    
    merged_df = merged_df.sort_values(by=['Country', 'Year'])
    
    print("\nFirst 10 rows of the final merged dataset:")
    print(merged_df.head(10))
    
    print("\nLast 10 rows of the final merged dataset:")
    print(merged_df.tail(10))
    
    print("\nDataset Info (columns, data types, nulls):")
    merged_df.info()

    output_filename = 'all_indicators_merged.csv'
    merged_df.to_csv(output_filename, index=False)
    print(f"\nSuccessfully saved merged data to {output_filename}")

else:
    print("Failed to load the first file. Stopping script.")