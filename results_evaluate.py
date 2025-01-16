import pandas as pd
import numpy as np
import re
import pprint
import joblib
from scipy.stats import spearmanr
from scipy.stats import kendalltau

import scipy.stats as stats
import os

#df = pd.read_csv("/home/oemerfar/omer_llm/all_results/results_no_bib_recrsv_chnk2048_ovl256_Meta-Llama-3.1-70B-AQLM-PV.csv")
#df = pd.read_csv("/home/oemerfar/omer_llm/all_results/results_no_bib_recrsv_chnk1024_ovl128_Meta-Llama-3.1-70B-AQLM-PV.csv")
#df = pd.read_csv("/home/oemerfar/omer_llm/all_results/results_no_bib_recrsv_chk1000_ovl200_Meta-Llama-3.1-70B-AQLM-PV.csv")
#df = pd.read_csv("/home/oemerfar/omer_llm/all_results/results_no_bib_nltk_chnk1000_ovl200_Meta-Llama-3.1-70B-AQLM-PV.csv")
#df = pd.read_csv("/home/oemerfar/omer_llm/all_results/new_results/combined_chunk_cot/results_combined_cot_chunk_512_ovl_20_Meta-Llama-3.1-70B-AQLM-PV.csv")
#df = pd.read_csv("/home/oemerfar/omer_llm/all_results/new_results/results_knowledge_augmented_chunk_512_ovl_20_Meta-Llama-3.1-70B-AQLM-PV.csv")
#df = pd.read_csv("/home/oemerfar/omer_llm/all_results/new_results/results_rcrsv_chunk_512_ovl_20_Meta-Llama-3.1-70B-AQLM-PV.csv")
#df = pd.read_csv("/home/oemerfar/omer_llm/all_results/new_results/results_semantic_chunk_512_ovl_20_Meta-Llama-3.1-70B-AQLM-PV.csv")
#df = pd.read_csv("/home/oemerfar/omer_llm/all_results/new_results/results_SciBERT_combined_chunk_512_ovl_20_Meta-Llama-3.1-70B-AQLM-PV.csv")
#df = pd.read_csv("/home/oemerfar/omer_llm/all_results/new_results/results_Llama3.1-8B_combined_chunk_512_ovl_20_Meta-Llama-3.1-70B-AQLM-PV.csv")
#df = pd.read_csv("/home/oemerfar/omer_llm/all_results/new_results/results_combined_chunk_512_ovp_20_base_fallback_with_recursive_Meta-Llama-3.1-70B-AQLM-PV-.csv")

df = pd.read_csv("/home/oemerfar/omer_llm/all_results/new_results/combined_chunk/results_combined_chunk_512_ovl_20_Meta-Llama-3.1-70B-AQLM-PV.csv")

#print("DATA AT THE BEGINNING:",len(df))

# Define the lists of documents for each topic
motor_imagery_docs = [
    '10.pdf', '34.pdf', '35.pdf', '36.pdf', '37.pdf',
    'MI1.pdf', 'MI2.pdf', 'MI3.pdf', 'MI4.pdf', 'MI5.pdf',
    'MI6.pdf', 'MI7.pdf', 'MI8.pdf', 'MI9.pdf', 'MI10.pdf'
]
auditory_attention_docs = [
    '38.pdf', '39.pdf', '40.pdf', '41.pdf', '42.pdf',
    '43.pdf', '44.pdf', '45.pdf', '46.pdf', '47.pdf',
    '48.pdf'
]
ie_attention_docs = [
    '17.pdf', '49.pdf', '50.pdf', '51.pdf', '52.pdf',
    '53.pdf', '54.pdf', '55.pdf', '56.pdf', '57.pdf'
]
# Define the electrode code to regions mappings
electrode_to_regions = {
    "Fz": ["Frontal"],
    "F1": ["Frontal"],
    "F2": ["Frontal"],
    "F3": ["Frontal"],
    "F4": ["Frontal"],
    "F5": ["Frontal"],
    "F6": ["Frontal"],
    "F7": ["Frontal"],
    "F8": ["Frontal"],
    "Fpz": ["Frontal"],
    "Fp1": ["Frontal"],
    "Fp2": ["Frontal"],
    "AFz": ["Frontal"],
    "AF3": ["Frontal"],
    "AF4": ["Frontal"],
    "AF7": ["Frontal"],
    "AF8": ["Frontal"],
    "FT7": ["Frontal, Temporal"],
    "FT8": ["Frontal, Temporal"],
    "FT9": ["Frontal, Temporal"],
    "FT10": ["Frontal, Temporal"],
    "Cz": ["Central", "Motor"],
    "C1": ["Central", "Motor"],
    "C2": ["Central", "Motor"],
    "C3": ["Central", "Motor"],
    "C4": ["Central", "Motor"],
    "C5": ["Central", "Motor"],
    "C6": ["Central", "Motor"],
    "CPz": ["Central", "Motor"],
    "CP1": ["Central", "Motor"],
    "CP2": ["Central", "Motor"],
    "CP3": ["Central", "Motor"],
    "CP4": ["Central", "Motor"],
    "CP5": ["Central", "Motor"],
    "CP6": ["Central", "Motor"],
    "FCz": ["Motor"],
    "FC1": ["Motor"],
    "FC2": ["Motor"],
    "FC3": ["Motor"],
    "FC4": ["Motor"],
    "FC5": ["Motor"],
    "FC6": ["Motor"],
    "Pz": ["Parietal"],
    "P1": ["Parietal"],
    "P2": ["Parietal"],
    "P3": ["Parietal"],
    "P4": ["Parietal"],
    "P5": ["Parietal"],
    "P6": ["Parietal"],
    "P7": ["Parietal"],
    "P8": ["Parietal"],
    "P9": ["Parietal"],
    "P10": ["Parietal"],
    "TP7": ["Parietal", "Temporal"],
    "TP8": ["Parietal", "Temporal"],
    "TP9": ["Parietal", "Occipital"],
    "TP10": ["Parietal", "Occipital"],
    "POz": ["Parietal", "Occipital"],
    "PO3": ["Parietal", "Occipital"],
    "PO4": ["Parietal", "Occipital"],
    "PO7": ["Parietal", "Occipital"],
    "PO8": ["Parietal", "Occipital"],
    "Oz": ["Occipital"],
    "O1": ["Occipital"],
    "O2": ["Occipital"],
    "T7": ["Temporal"],
    "T8": ["Temporal"],
    "FT7": ["Temporal"],
    "FT8": ["Temporal"],
    "FT9": ["Frontal"],
    "FT10": ["Frontal"],
}

# Function to split rows with multiple frequencies for a single region
def expand_multiple_frequencies(df):
    # Split each row into multiple rows if 'Frequency Band' contains multiple frequencies separated by commas
    df = df.assign(
        **{'Frequency Band': df['Frequency Band'].str.split(',\s*')}
    ).explode('Frequency Band').reset_index(drop=True)
    return df

df_incorrect = df.copy()

# Initialize the 'Topic' column with NaN
df_incorrect['Topic'] = np.nan

# Assign topics to the relevant documents
df_incorrect.loc[df_incorrect['Document'].isin(motor_imagery_docs), 'Topic'] = 'MI'
df_incorrect.loc[df_incorrect['Document'].isin(auditory_attention_docs), 'Topic'] = 'AA'
df_incorrect.loc[df_incorrect['Document'].isin(ie_attention_docs), 'Topic'] = 'IEA'

# Expand multiple frequencies into separate rows before processing
df_incorrect = expand_multiple_frequencies(df_incorrect)

# Define the valid frequency bands and regions
valid_frequency_bands = ['Alpha', 'Beta', 'Delta', 'Theta', 'Gamma']
valid_regions = ['Frontal', 'Motor', 'Central', 'Temporal', 'Parietal', 'Occipital']

# Filter the DataFrame
df_main_correct = df_incorrect[
    df_incorrect['Frequency Band'].isin(valid_frequency_bands) & 
    df_incorrect['Region'].isin(valid_regions)
]

#print("CORRECT DATA AT THE BEGINNING", (len(df_main_correct)))
#print("\nDF MAIN CORRECT\n", df_main_correct)

filtered_df = df_incorrect[
    ~df_incorrect['Frequency Band'].isin(valid_frequency_bands) | 
    ~df_incorrect['Region'].isin(valid_regions)
]

# Display the filtered DataFrame
print("WRONG FORMAT DATA SIZE",len(filtered_df))
#print("Data before cleaning wrong formats: \n",filtered_df)

print("\n----------------------------------------------------------------------------------------")

def standardize_region(region_str, electrode_map, valid_regions):
    if pd.isnull(region_str):
            return np.nan
        
    # Initialize a set to store found regions
    found_regions = set()
        
    # Step 1: Substring matching with valid regions
    region_str_lower = region_str.lower()
    for valid_region in valid_regions:
        if valid_region.lower() in region_str_lower:
            found_regions.add(valid_region)
                
    # Convert to string and strip whitespace
    region_str = str(region_str).strip()
    
    # Step 2: Check if the entire string is an electrode code
    if region_str in electrode_map:
        found_regions.update(electrode_map[region_str])
    
    # Step 3: Check if the region_str contains any electrode codes
    tokens = re.findall(r'\b\w+\b', region_str)
    for token in tokens:
        if token in electrode_map:
            found_regions.update(electrode_map[token])
    
    # Step 3: Check if the region_str is already a valid region
    if region_str in valid_regions:
        found_regions.add(region_str)
    
    if found_regions:
        return list(found_regions)
    else:
        return np.nan
        
'''def resolve_regions(row):
    regions = row['Region']
    topic = row['Topic']
    
    if isinstance(regions, list):
        if topic == 'MI':
            # Define priority rules
            # For Central and Motor overlap
            if 'Central' in regions and 'Motor' in regions:
                regions = ['Motor']
            # For Parietal and Temporal overlap
            if 'Parietal' in regions and 'Temporal' in regions:
                regions = ['Parietal']
            # For Parietal and Occipital overlap
            if 'Parietal' in regions and 'Occipital' in regions:
                regions = ['Parietal']
            # For Frontal and Temporal overlap
            if 'Frontal' in regions and 'Temporal' in regions:
                regions = ['Frontal']
        
        elif topic == 'AA':
            if 'Central' in regions and 'Motor' in regions:
                regions = ['Motor']
            if 'Parietal' in regions and 'Temporal' in regions:
                regions = ['Temporal']
            if 'Parietal' in regions and 'Occipital' in regions:
                regions = ['Occipital']
            if 'Frontal' in regions and 'Temporal' in regions:
                regions = ['Temporal']
        
        elif topic == 'IEA':
            if 'Central' in regions and 'Motor' in regions:
                regions = ['Central']
            if 'Parietal' in regions and 'Temporal' in regions:
                regions = ['Parietal']
            if 'Frontal' in regions and 'Temporal' in regions:
                regions = ['Frontal']
            if 'Parietal' in regions and 'Occipital' in regions:
                # Duplicate row: one with Parietal, one with Occipital
                return ['Parietal', 'Occipital']
        
        # If only one region or no overlapping, retain as is
        return regions
    else:
        return regions  # Return as is if not a list'''
    
'''def duplicate_rows(df):
    # List to collect new rows
    new_rows = []
    
    # Iterate over the DataFrame rows
    for idx, row in df.iterrows():
        regions = row['Region']
        if isinstance(regions, list) and len(regions) > 1:
            # Create a new row for each region
            for region in regions:
                new_row = row.copy()
                new_row['Region'] = region
                new_rows.append(new_row)
            # Remove the original row
            df.drop(idx, inplace=True)
    
    # Append the new rows
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
    
    return df'''

def flatten_region(region):
    if pd.isnull(region):
        return np.nan
    elif isinstance(region, list):
        if len(region) == 1:
            return region[0]
        else:
            # Handle unexpected multi-element lists
            return ', '.join(region)
    else:
        return region

def standardize_frequency(freq_str, valid_freqs):
    if pd.isnull(freq_str):
        return np.nan

    # Convert to string and strip whitespace
    freq_str = str(freq_str).strip()

    # Step 1: Replace 'Mu' and its variations with 'Alpha'
    if re.fullmatch(r'(?i)mu', freq_str):
        return 'Alpha'
    if re.fullmatch(r'(?i)mu band', freq_str):
        return 'Alpha'

    # Step 2: Convert Hz values to frequency names
    if 'hz' in freq_str.lower():
        # Extract the numeric value
        match = re.search(r'([\d\.]+)\s*hz', freq_str, re.IGNORECASE)
        if match:
            freq_val = float(match.group(1))
            if 0 < freq_val < 4:
                return 'Delta'
            elif 4 <= freq_val <= 8:
                return 'Theta'
            elif 8 < freq_val <= 12:
                return 'Alpha'
            elif 12 < freq_val <= 30:
                return 'Beta'
            elif freq_val > 30:
                return 'Gamma'
            else:
                return np.nan  # Frequency out of expected ranges
        else:
            return np.nan  # Unable to extract numeric value

    # Step 3: Standardize entries with additional text
    for freq in valid_freqs:
        # Use word boundaries to avoid partial matches
        pattern = r'(?i)\b' + re.escape(freq) + r'\b'
        if re.search(pattern, freq_str):
            return freq

    # If none of the above conditions are met, return NaN
    return np.nan


# Apply the mapping function to the 'Region' column
filtered_df['Region'] = filtered_df['Region'].apply(
    lambda x: standardize_region(x, electrode_to_regions, valid_regions)
)

#print("DataFrame after mapping regions:\n")
#print(filtered_df)

print("\n----------------------------------------------------------------------------------------")

# Apply the resolve_regions function
#filtered_df['Region'] = filtered_df.apply(resolve_regions, axis=1)

# Replace NaN in 'Mapped Regions' with an empty list to avoid issues with explode
filtered_df['Region'] = filtered_df['Region'].apply(lambda x: x if isinstance(x, list) else [])

# Explode the 'Mapped Regions' list into separate rows
filtered_df = filtered_df.explode('Region')

# Display the DataFrame after resolving regions
#print("\nDataFrame after resolving regions:")
#print(filtered_df)

print("\n----------------------------------------------------------------------------------------")

# Apply the duplication and flatten function
#filtered_df = duplicate_rows(filtered_df)
filtered_df['Region'] = filtered_df['Region'].apply(flatten_region)
#print(filtered_df)

print("\n----------------------------------------------------------------------------------------")

# Optional: Verify the mapping by filtering out NaN entries
invalid_regions = filtered_df[filtered_df['Region'].isna()]
if not invalid_regions.empty:
    print(f"{len(invalid_regions)} entries couldn't be mapped to valid regions:")
    #print(invalid_regions)
else:
    print("All regions have been successfully mapped to valid regions.")

print("\n----------------------------------------------------------------------------------------")

df_cleaned = filtered_df.copy()

# Drop rows where 'Region' is NaN
df_result = df_cleaned.dropna(subset=['Region'])
df_result.reset_index(drop=True, inplace=True)
#print("Cleaned Regions Data",df_result)
print("\n----------------------------------------------------------------------------------------")

df_frequency = df_result.copy()
# Display values before cleaning
print("Values in 'Frequency Band' before cleaning:")
#print(df_frequency)
print("\n----------------------------------------------------------------------------------------")
df_cleaned_fq = df_frequency.copy()

# Apply the cleaning function
df_cleaned_fq['Frequency Band'] = df_frequency['Frequency Band'].apply(lambda x: standardize_frequency(x, valid_frequency_bands))

invalid_fq = df_cleaned_fq[df_cleaned_fq['Frequency Band'].isna()]
print(f"{len(invalid_fq)} entries couldn't be mapped to valid frequency band:")
#print(invalid_fq)
print("\n----------------------------------------------------------------------------------------")

# Drop rows where 'Frequency Band' is NaN
df_result_all = df_cleaned_fq.dropna(subset=['Frequency Band'])
df_result_all.reset_index(drop=True, inplace=True)

# Display values after cleaning
print("\nValues in 'Frequency Band Cleaned' after cleaning:")
#print("DF_RESULT_ALL",df_result_all)
#print("DF_MAIN_CORRECT",df_main_correct)
print("DF_MAIN_CORRECT SIZE",len(df_main_correct))

print("\n----------------------------------------------------------------------------------------")

df_final_results = pd.concat([df_result_all, df_main_correct])
print("DF_FINAL_RESULTS SIZE:", len(df_final_results))

#print(df_final_results.loc[df_final_results['Document'] == "MI1.pdf"])
topic_regions = {
    'MI': ['Frontal', 'Motor', 'Parietal'],
    'AA': ['Frontal', 'Temporal', 'Occipital'],
    'IEA': ['Frontal', 'Occipital', 'Parietal', 'Central']
}

# Filter the DataFrame based on the topic and its valid regions
#df_final_results = df_final_results[df_final_results.apply(lambda x: x['Region'] in topic_regions[x['Topic']], axis=1)]

print("\n----------------------------------------------------------------------------------------")
#unique_combinations = df_final_results.drop_duplicates(subset=['Topic', 'Frequency Band', 'Region'])
#print("DF FINAL RESULTS\n",df_final_results[(df_final_results['Topic']=='MI') & (df_final_results['Frequency Band'] == 'Alpha')])


# Step 1: Aggregate Data
aggregated_data = df_final_results.groupby(['Topic', 'Frequency Band', 'Region']).size().reset_index(name='Count')
topics = aggregated_data['Topic'].unique()
full_index = pd.MultiIndex.from_product([topics, valid_frequency_bands, valid_regions], names=['Topic', 'Frequency Band', 'Region'])
full_df = pd.DataFrame(index=full_index).reset_index()

#print("AGGREGATED DATA\n",aggregated_data)
# Step 4: Merge full_df with aggregated_data and filter based on topic_regions
full_df = full_df.merge(aggregated_data, on=['Topic', 'Frequency Band', 'Region'], how='left')
full_df['Count'].fillna(0, inplace=True)

# Filter and rank data
'''filtered_data = pd.concat([
    full_df[(full_df['Topic'] == topic) & (full_df['Region'].isin(regions))]
    for topic, regions in topic_regions.items()
])'''
filtered_data = full_df
print("DF FILTERED DATA\n",filtered_data[(filtered_data['Topic']=='AA')])
#print("DF FILTERED DATA BEFORE RANKING\n",filtered_data)

# Apply ranking
filtered_data['Rank'] = filtered_data.groupby(['Topic', 'Frequency Band'])['Count'].rank(method='average', ascending=False, na_option='bottom')
filtered_data.sort_values(by=['Topic', 'Frequency Band', 'Rank'], inplace=True)

#print("DF FILTERED DATA\n",filtered_data[(filtered_data['Topic']=='AA') & (filtered_data['Frequency Band'] == 'Alpha')])

# Post-processing to ensure correct ranking for zero data
'''for (topic, band), group in filtered_data.groupby(['Topic', 'Frequency Band']):
    if group['Count'].sum() == 0:  # Check if there's no data for this band
        filtered_data.loc[group.index, 'Rank'] = 1  # Assign rank 1 to all regions
    else:
        max_rank = group['Rank'].max()
        zero_data_indices = group[group['Count'] == 0].index
        filtered_data.loc[zero_data_indices, 'Rank'] = max_rank'''

# Step 6: Organize results into a structured format
formatted_results = {}
for name, group in filtered_data.groupby('Topic'):
    formatted_results[name] = {}
    for band, band_group in group.groupby('Frequency Band'):
        formatted_results[name][band] = dict(zip(band_group['Region'], band_group['Rank']))

print("FORMATTED RESULTS LLM\n")
pprint.pprint(formatted_results)

# Load the dictionary from the file
hendrik_xai_result = joblib.load('/home/oemerfar/omer_llm/hendrik_xai_result/hendrik_xai_results_dictionary')

#print("HENDRIKS RESULTS:\n")
#pprint.pprint(hendrik_xai_result)

# Mapping subregions to main regions

regions_map = {
    'Frontal': ['Left Frontal', 'Right Frontal'],
    'Motor': ['Left Motor', 'Right Motor'],
    'Parietal': ['Left Parietal', 'Right Parietal'],
    'Temporal': ['Left Temporal', 'Right Temporal'],
    'Occipital': ['Left Occipital', 'Right Occipital'],
    'Central': ['Central']
}

# Function to calculate averages and keep existing main regions unchanged
def average_regions_to_new_dict(data, regions_map):
    averaged_data = {}
    
    # Loop through each task (e.g., 'IEA', 'AA', 'MI')
    for task, sides in data.items():
        averaged_data[task] = {}
        
        # Loop through each task type (e.g., 'Ext.', 'Int.')
        for task_type, freqs in sides.items():
            if task_type not in averaged_data[task]:
                averaged_data[task][task_type] = {}
            
            # Loop through each frequency band (e.g., 'alpha', 'beta')
            for freq_band, regions in freqs.items():
                if freq_band not in averaged_data[task][task_type]:
                    averaged_data[task][task_type][freq_band] = {}
                    
                # Loop through each region in the current data
                for region, value in regions.items():
                    # If the region is already a main region, copy it unchanged
                    if region not in [subregion for subregions in regions_map.values() for subregion in subregions]:
                        averaged_data[task][task_type][freq_band][region] = value
                
                # Loop through the subregions that need to be averaged
                for main_region, subregions in regions_map.items():
                    subregion_values = [regions.get(subregion, 0) for subregion in subregions if subregion in regions]
                    
                    # Only average if there are valid subregions found
                    if subregion_values:
                        avg_value = sum(subregion_values) / len(subregion_values)
                        averaged_data[task][task_type][freq_band][main_region] = avg_value

    return averaged_data

# Calling the function to create a new dictionary with averaged main regions
hendrik_xai_result_main_regions = average_regions_to_new_dict(hendrik_xai_result, regions_map)

#print("\nHENDRIKS RESULTS MAIN REGIONS:\n")
#pprint.pprint(hendrik_xai_result_main_regions)

# Function to calculate the average of left-right or int-ext sides
def merge_sides_and_average(data):
    merged_data = {}

    # Loop through each task (e.g., 'AA', 'IEA', 'MI')
    for task, sides in data.items():
        merged_data[task] = {}
        
        # Loop through each side (e.g., 'Left', 'Right', 'Ext.', 'Int.')
        for side_name, freqs in sides.items():
            # Loop through each frequency band (e.g., 'alpha', 'beta')
            for freq_band, regions in freqs.items():
                if freq_band not in merged_data[task]:
                    merged_data[task][freq_band] = {}

                # Loop through each region in the frequency band
                for region, value in regions.items():
                    # If the region is already in the new dict, calculate the average
                    if region in merged_data[task][freq_band]:
                        merged_data[task][freq_band][region] = (
                            merged_data[task][freq_band][region] + value
                        ) / 2
                    else:
                        merged_data[task][freq_band][region] = value

    return merged_data
    
hendrik_xai_result_no_sides = merge_sides_and_average(hendrik_xai_result_main_regions)

#print("\nHENDRIKS RESULTS NO SIDES:\n")
#pprint.pprint(hendrik_xai_result_no_sides)

# Function to convert values to rankings within each frequency band
'''def convert_values_to_rankings(data):
    ranked_data = {}

    # Loop through each task (e.g., 'AA', 'IEA', 'MI')
    for task, freqs in data.items():
        ranked_data[task] = {}

        # Loop through each frequency band (e.g., 'alpha', 'beta')
        for freq_band, regions in freqs.items():
            # Sort regions by value and assign rankings
            sorted_regions = sorted(regions.items(), key=lambda x: x[1], reverse=True)
            rankings = {region: rank + 1 for rank, (region, value) in enumerate(sorted_regions)}
            
            # Add rankings to the new dictionary
            ranked_data[task][freq_band] = rankings

    return ranked_data

hendrik_xai_result_ranked = convert_values_to_rankings(hendrik_xai_result_no_sides)
print("\nHENDRIKS RESULTS RANKED:\n")
pprint.pprint(hendrik_xai_result_ranked)'''

def convert_values_to_rankings_average_method(data):
    ranked_data = {}

    # Loop through each task (e.g., 'AA', 'IEA', 'MI')
    for task, freqs in data.items():
        ranked_data[task] = {}

        # Loop through each frequency band (e.g., 'alpha', 'beta')
        for freq_band, regions in freqs.items():
            # Extract region names and their values
            region_names, values = zip(*regions.items())
            
            # Calculate rankings using the 'average' method for ties
            rankings = stats.rankdata([-v for v in values], method='average')  # '-v' for descending order
            
            # Map the rankings back to the region names
            ranked_regions = {region: rank for region, rank in zip(region_names, rankings)}
            
            # Add the ranked regions to the new dictionary
            ranked_data[task][freq_band] = ranked_regions

    return ranked_data

# Call the function to convert values to rankings using the average method
hendrik_xai_result_ranked = convert_values_to_rankings_average_method(hendrik_xai_result_no_sides)

print("\nHENDRIKS RESULTS RANKED:\n")
pprint.pprint(hendrik_xai_result_ranked)

# Convert frequency band names in hendrik_xai_result_ranked to match formatted_results
hendrik_xai_result_ranked_standardized = {}
for topic in hendrik_xai_result:
    hendrik_xai_result_ranked_standardized[topic] = {}
    for freq_band in hendrik_xai_result_ranked[topic]:
        # Standardize frequency band name
        freq_band_standardized = freq_band.capitalize()
        hendrik_xai_result_ranked_standardized[topic][freq_band_standardized] = hendrik_xai_result_ranked[topic][freq_band]

# Standardize frequency band names in Hendrik's data
def standardize_frequency_band_names(data):
    standardized_data = {}
    for task, freqs in data.items():
        standardized_data[task] = {}
        for freq_band, regions in freqs.items():
            # Capitalize the first letter
            freq_band_standardized = freq_band.capitalize()
            standardized_data[task][freq_band_standardized] = regions
    return standardized_data

hendrik_xai_result_no_sides_standardized = standardize_frequency_band_names(hendrik_xai_result_no_sides)
print("\nHENDRIKS RESULTS STANDARDIZED:\n")
pprint.pprint(hendrik_xai_result_ranked_standardized)

spearman_results = {}

for topic in formatted_results.keys():
    spearman_results[topic] = {}
    print(f"\nTopic: {topic}")
    
    for freq_band in formatted_results[topic].keys():
        # Get the regions from both datasets
        regions_llm = set(formatted_results[topic][freq_band].keys())
        regions_hendrik = set(hendrik_xai_result_ranked_standardized[topic][freq_band].keys())
        common_regions = regions_llm.intersection(regions_hendrik)
        
        if not common_regions:
            print(f"  Frequency Band: {freq_band}, No common regions to compare.")
            continue
        
        # Extract the ranks for the common regions
        ranks_llm = [formatted_results[topic][freq_band][region] for region in common_regions]
        ranks_hendrik = [hendrik_xai_result_ranked_standardized[topic][freq_band][region] for region in common_regions]
        
        # Compute Spearman's rho
        rho, p_value = spearmanr(ranks_llm, ranks_hendrik)
        
        # Store the results
        spearman_results[topic][freq_band] = {'rho': rho, 'p_value': p_value}
        
        # Print the results
        print(f"  Frequency Band: {freq_band}, Spearman's rho: {rho:.3f}, p-value: {p_value:.3f}")

print("\nSpearman's Rank Correlation Results:")
pprint.pprint(spearman_results)

# Initialize a dictionary to store Kendall's tau results
kendall_results = {}

for topic in formatted_results.keys():
    kendall_results[topic] = {}
    print(f"\nTopic: {topic}")
    
    for freq_band in formatted_results[topic].keys():
        # Get the regions from both datasets
        regions_llm = set(formatted_results[topic][freq_band].keys())
        regions_hendrik = set(hendrik_xai_result_ranked_standardized[topic][freq_band].keys())
        common_regions = regions_llm.intersection(regions_hendrik)
        
        if not common_regions:
            print(f"  Frequency Band: {freq_band}, No common regions to compare.")
            continue
        
        # Extract the ranks for the common regions
        ranks_llm = [formatted_results[topic][freq_band][region] for region in common_regions]
        ranks_hendrik = [hendrik_xai_result_ranked_standardized[topic][freq_band][region] for region in common_regions]
        
        # Compute Kendall's Tau
        tau, p_value = kendalltau(ranks_llm, ranks_hendrik)
        
        # Store the results
        kendall_results[topic][freq_band] = {'tau': tau, 'p_value': p_value}
        
        # Print the results
        print(f"  Frequency Band: {freq_band}, Kendall's tau: {tau:.3f}, p-value: {p_value:.3f}")

# Optionally, print the full results
print("\nKendall's Tau Correlation Results:")
pprint.pprint(kendall_results)