import zipfile
import os
import pandas as pd

# Define file paths as raw strings
directory_path = r'C:\Users\Thanhnt\Downloads\PV_Forecasting\DataLoader\Nrel\Rawdata'
output_path = r'C:\Users\Thanhnt\Downloads\PV_Forecasting\DataLoader\Nrel\data'

file_names = os.listdir(directory_path)
file_names = [f for f in file_names if os.path.isfile(os.path.join(directory_path, f))]

for i, file_name in enumerate(file_names):
    df_2021 = pd.DataFrame()
    df_2022 = pd.DataFrame()
    
    with zipfile.ZipFile(os.path.join(directory_path, file_name), 'r') as zip_ref:
        # List all files in the ZIP
        files_in_zip = zip_ref.namelist()
        
        # Loop through each CSV file and read it into a pandas DataFrame
        for csv_file in files_in_zip:
            with zip_ref.open(csv_file) as file:
                df = pd.read_csv(file)
                position = df.iloc[0:1][['Latitude','Longitude']]
                latitude = position['Latitude'].values[0]
                longitude = position['Longitude'].values[0]
                
                n = len(df) - 2
                data_po = {'Latitude': [latitude] * n, 'Longitude': [longitude] * n}
                position = pd.DataFrame(data_po)
                
                second_dataset = df.iloc[1:].reset_index(drop=True)
                new_header = df.iloc[1]  # The second row to be the new header
                second_dataset = df[2:].reset_index(drop=True)
                second_dataset.columns = new_header
                second_dataset = second_dataset.apply(pd.to_numeric, errors='coerce')
                second_dataset = second_dataset[['Year','Month','Day','Hour','Minute','Temperature','GHI','Relative Humidity']]
                
                data = pd.concat([second_dataset, position], axis=1)
                
            if '_2021' in csv_file:
                df_2021 = pd.concat([df_2021, data])
                name = 'data2021_' + str(i) + '.csv'
                df_2021.to_csv(os.path.join(output_path, name))
                
            if '_2022' in csv_file:
                df_2022 = pd.concat([df_2022, data])
                name = 'data2022_' + str(i) + '.csv'
                df_2022.to_csv(os.path.join(output_path, name))
