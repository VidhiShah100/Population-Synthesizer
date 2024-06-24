# importing the necessary libraries
import pandas as pd
import numpy as np

# reading the Data.csv file into a Pandas dataframe
df = pd.read_csv(r'Data.csv')
df.rename(columns={'Age_category': 'Age_group'}, inplace=True)

# Specifying frequencies of each category of each feature as required
desired_frequencies = {
    'Sex': {1: 25324, 2: 24676},
    'Age_group': {1: 17955, 2: 29642, 3: 2403},
    'Highest_education_level': {0: 7490, 1: 5655, 2: 24400, 3: 12455}
}

# Converting dataframe to numpy array
data = df.values

synthetic_data = []

for col_index, feature in enumerate(df.columns):
    feature_data = data[:, col_index]  # Extracting the feature data
    unique_values, counts = np.unique(feature_data, return_counts=True)
    current_frequencies = dict(zip(unique_values, counts))

    for value in desired_frequencies[feature]:
        # calculating differnece between number of samples already present v/s number of samples needed to decide 
        # whether to perform oversampling or undersampling
        diff = desired_frequencies[feature][value] - current_frequencies.get(value, 0)  
        if diff > 0:  # Oversampling being performed
            indices = np.where(feature_data == value)[0]
            oversampled_indices = np.random.choice(indices, size=diff, replace=True)
            synthetic_data.extend(data[oversampled_indices])
        elif diff < 0:  # Undersampling being performed
            indices = np.where(feature_data == value)[0]
            undersampled_indices = np.random.choice(indices, size=-diff, replace=False)
            feature_data = np.delete(feature_data, undersampled_indices)
            data = np.delete(data, undersampled_indices, axis=0)


# Shuffling the synthetic data
np.random.shuffle(synthetic_data)
# Convert the list to a dataframe containing 50000 samples
synthetic_df = pd.DataFrame(synthetic_data[:50000], columns=df.columns)

# Save the new dataframe formed to a .csv file
synthetic_df.to_csv('synthetic_dataset.csv', index=False)

# Mapping the data to their respective categorical representation as specified
sex_mapping = {1: 'Male', 2: 'Female'}
age_group_mapping = {1: 'Below 22 years', 2: '22-60 years', 3: 'Above 60 years'}
education_mapping = {0: 'No formal education', 1: 'Primary Education', 2: 'Secondary Education', 3: 'Graduation and above'}

# Calculating frequencies of each category of each feature
sex_freq = synthetic_df['Sex'].map(sex_mapping).value_counts().sort_index()
age_freq =synthetic_df['Age_group'].map(age_group_mapping).value_counts().sort_index()
education_freq = synthetic_df['Highest_education_level'].map(education_mapping).value_counts().sort_index()

# tabs list created for ease of formatting .txt file
tabs = []
for i in range(3,6):
    tabs.append('\t'*(2*i))

# writing freqeuncies to a .txt file as required
with open('frequencies.txt', 'w') as f:
    # Header
    f.write('Variable'+tabs[1]+'Description'+tabs[1]+'Frequency\n')
    f.write('--------------------------------------------------------------------------------------------\n')
 
    # Writing data for Sex
    f.write('Sex\n')
    for description, count in sex_freq.items():
        f.write(tabs[2]+f'{description}'+tabs[1]+f'\t{count}\n')
    f.write('--------------------------------------------------------------------------------------------\n')

    # Writing data for Age_group
    f.write('Age_group\n')
    for description, count in age_freq.items():
        f.write(tabs[2]+f'{description}'+tabs[1]+f'{count}\n')
    f.write('--------------------------------------------------------------------------------------------\n')

    # Writing data for Highest_education_level
    f.write('Highest_education_level\n')
    for description, count in education_freq.items():
        f.write(tabs[2]+f'{description}'+tabs[0]+f'{count}\n')


# comparing statistical properties of the original and synthetic dataset to observe their similarity
print("Original Dataset Summary Statistics:")
orig_stats = df.describe()
print(orig_stats)

print("\nSynthetic Dataset Summary Statistics:")
syn_stats = synthetic_df.describe()
print(syn_stats)
