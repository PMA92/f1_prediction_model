#get for each driver over each year for each race, their position and keep it in an average. 
import pandas as pd

def avg_positions(year):
    for year in range(2022, 2025):
        data = pd.read_excel(f'data/data_{year}.xlsx')
        data.shift(1)
        avg_pos = data.groupby('Abbreviation')['Position'].mean()
        avg_pos.to_excel(f'data/avg_positions_{year}.xlsx', index=True)
