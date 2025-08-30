#!/usr/bin/env python
"""
Skrypt do uzupełnienia wszystkich danych (ceny metali, wskaźniki makroekonomiczne, opóźnienia)
dla pełnego zbioru danych do lipca 2025.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def complete_dataset():
    """
    Uzupełnia brakujące dane w zbiorze danych do lipca 2025
    """
    # Ścieżka do pliku z danymi
    file_path = "data/processed/processed_data_raw.csv"
    
    # Sprawdź czy plik istnieje
    if not os.path.exists(file_path):
        print(f"Błąd: Plik {file_path} nie istnieje!")
        return
    
    # Wczytaj istniejące dane
    data = pd.read_csv(file_path)
    print(f"Wczytano {len(data)} wierszy z istniejącego zbioru danych.")
    
    # Konwertuj datę do formatu datetime
    data['data'] = pd.to_datetime(data['data'])
    
    # Sprawdź zakres dat
    min_date = data['data'].min()
    max_date = data['data'].max()
    print(f"Zakres dat: od {min_date.strftime('%Y-%m-%d')} do {max_date.strftime('%Y-%m-%d')}")
    
    # Sprawdź brakujące wartości przed uzupełnieniem
    print("\nBrakujące wartości przed uzupełnieniem:")
    missing_values = data.isnull().sum()
    for col in data.columns:
        if missing_values[col] > 0:
            print(f"{col}: {missing_values[col]} ({missing_values[col]/len(data):.2%})")
    
    # Uzupełnij brakujące wartości dla opóźnionych cen metali (lag)
    metals = ["Złoto", "Srebro", "Platyna", "Pallad", "Miedź"]
    
    # 1. Uzupełnij wartości lag_1 i lag_3 dla wszystkich metali
    for metal in metals:
        # Sprawdź czy kolumny lag_1 i lag_3 istnieją
        lag1_col = f'{metal}_lag_1'
        lag3_col = f'{metal}_lag_3'
        
        # Uzupełnij lag_1
        data[lag1_col] = data[metal].shift(1)
        
        # Uzupełnij lag_3
        data[lag3_col] = data[metal].shift(3)
    
    # 2. Uzupełnij wskaźniki makroekonomiczne używając forward fill dla najnowszych danych
    # Lista kolumn makroekonomicznych
    macro_cols = ['inflacja', 'stopy_procentowe', 'bezrobocie', 'pkb', 'pkb_global', 
                 'kurs_usd', 'indeks_vix']
    
    # Jeśli wartości makro są brakujące w najnowszych danych, uzupełnij je ostatnimi dostępnymi
    for col in macro_cols:
        if data[col].tail(6).isnull().any():
            # Znajdź ostatnią dostępną wartość
            last_valid = data[col].iloc[:-6].dropna().iloc[-1]
            # Uzupełnij brakujące wartości
            data[col] = data[col].fillna(method='ffill')
            print(f"Uzupełniono brakujące wartości dla {col} używając ostatniej wartości: {last_valid}")
    
    # Sprawdź ponownie brakujące wartości
    print("\nBrakujące wartości po uzupełnieniu:")
    missing_values = data.isnull().sum()
    has_missing = False
    for col in data.columns:
        if missing_values[col] > 0:
            has_missing = True
            print(f"{col}: {missing_values[col]} ({missing_values[col]/len(data):.2%})")
    
    if not has_missing:
        print("Brak brakujących wartości! Dane są kompletne.")
    
    # Konwertuj datę z powrotem do formatu YYYY-MM-DD
    data['data'] = data['data'].dt.strftime('%Y-%m-%d')
    
    # Zapisz zaktualizowane dane
    data.to_csv(file_path, index=False)
    print(f"\nZaktualizowany zbiór danych został zapisany w {file_path}.")
    print(f"Łączna liczba wierszy: {len(data)}")
    print(f"Zakres dat: od {min_date.strftime('%Y-%m-%d')} do {max_date.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    complete_dataset()
