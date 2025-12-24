import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def load_data(path):
    return pd.read_csv(path, sep=';', decimal=',', header=0)

def preprocessing_data(df):

    df_prep = df.copy()

    # Drop kolom tidak berguna
    df_prep.drop(columns=['Unnamed: 15', 'Unnamed: 16'], errors='ignore', inplace=True)

    # Handle missing Date & Time
    df_prep['Date'] = df_prep['Date'].ffill()
    df_prep['Time'] = df_prep['Time'].ffill()

    # Gabungkan Date & Time
    df_prep['Datetime'] = pd.to_datetime(
        df_prep['Date'] + ' ' + df_prep['Time'],
        format='%d/%m/%Y %H.%M.%S',
        errors='coerce'
    )

    df_prep.drop(columns=['Date', 'Time'], inplace=True)

    # Handle missing values numerik
    num_cols = df_prep.select_dtypes(include='number').columns
    df_prep[num_cols] = df_prep[num_cols].fillna(df_prep[num_cols].mean())

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_prep[num_cols])

    df_scaled = pd.DataFrame(scaled_data, columns=num_cols)
    df_scaled['Datetime'] = df_prep['Datetime'].values

    return df_scaled

def save_data(df, output_path):
    df.to_csv(output_path, index=False)

def main():
    input_path = '../data/air_quality/airquality.csv'
    output_path = '../preprocessing/airquality_preprocessing.csv'

    df = load_data(input_path)
    df_clean = preprocessing_data(df)
    save_data(df_clean, output_path)

    print('Preprocessing selesai. File tersimpan:', output_path)

if __name__ == '__main__':
    main()
