#!/usr/bin/env python
"""
Skrypt do pobierania najnowszych cen metali z Yahoo Finance
i dodania ich do istniejącego zbioru danych do lipca 2025.
"""

import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import os

def get_financial_data(ticker, start_date, end_date):
    """
    Pobiera dane finansowe z Yahoo Finance.
    
    Args:
        ticker (str): Symbol instrumentu finansowego
        start_date (str): Data początkowa w formacie YYYY-MM-DD
        end_date (str): Data końcowa w formacie YYYY-MM-DD
        
    Returns:
        pd.DataFrame: DataFrame z danymi finansowymi
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval="1mo")
        if data.empty:
            print(f"Brak danych dla {ticker}")
            return None
            
        data.reset_index(inplace=True)
        data = data[["Date", "Close"]]
        data.columns = ["data", ticker]
        
        # Konwertuj datę na format YYYY-MM-DD (pierwszy dzień miesiąca)
        data["data"] = pd.to_datetime(data["data"]).dt.strftime("%Y-%m-01")
        
        return data
    except Exception as e:
        print(f"Błąd podczas pobierania {ticker}: {e}")
        return None

def get_metal_prices(start_date, end_date):
    """
    Pobiera historyczne ceny metali z Yahoo Finance.
    
    Metale:
    - Złoto (GC=F, XAUUSD=X, GLD)
    - Srebro (SI=F, XAGUSD=X, SLV)
    - Platyna (PL=F, PPLT)
    - Pallad (PA=F, PALL)
    - Miedź (HG=F, CPER)
    
    Args:
        start_date (str): Data początkowa w formacie YYYY-MM-DD
        end_date (str): Data końcowa w formacie YYYY-MM-DD
        
    Returns:
        pd.DataFrame: DataFrame z cenami metali
    """
    # Lista symboli metali (główne i zapasowe)
    metals = {
        ("GC=F", "XAUUSD=X", "GLD"): "Złoto",
        ("SI=F", "XAGUSD=X", "SLV"): "Srebro",
        ("PL=F", "PPLT"): "Platyna",
        ("PA=F", "PALL"): "Pallad",
        ("HG=F", "CPER"): "Miedź"
    }
    
    # Pobierz dane dla każdego metalu
    all_data = None
    for symbols, name in metals.items():
        print(f"Pobieranie danych dla {name}...")
        
        # Spróbuj pobrać dane z każdego symbolu po kolei
        df = None
        for symbol in symbols:
            try:
                df = get_financial_data(symbol, start_date, end_date)
                if df is not None and not df.empty:
                    # Sprawdź czy mamy wystarczająco dużo danych
                    if len(df) > 0:
                        print(f"Sukces! Pobrano {len(df)} rekordów z {symbol}")
                        break
                    else:
                        print(f"Brak danych dla {symbol}, próbuję następny symbol...")
                        df = None
            except Exception as e:
                print(f"Błąd podczas pobierania {symbol}: {e}")
                continue
        
        if df is not None and not df.empty:
            # Zmień nazwę kolumny na nazwę metalu
            df = df.rename(columns={next(col for col in df.columns if col != "data"): name})
            
            if all_data is None:
                all_data = df
            else:
                all_data = all_data.merge(df, on="data", how="outer")
        else:
            print(f"Nie udało się pobrać danych dla {name}")
    
    # Sortuj po dacie
    if all_data is not None:
        all_data = all_data.sort_values("data")
    
    return all_data

def add_lag_features(df):
    """
    Dodaje opóźnione wartości cen metali.
    
    Args:
        df (pd.DataFrame): DataFrame z cenami metali
        
    Returns:
        pd.DataFrame: DataFrame z dodanymi opóźnionymi wartościami
    """
    metals = ["Złoto", "Srebro", "Platyna", "Pallad", "Miedź"]
    result_df = df.copy()
    
    # Dodaj opóźnione wartości dla każdego metalu
    for metal in metals:
        result_df[f'{metal}_lag_1'] = result_df[metal].shift(1)
        result_df[f'{metal}_lag_3'] = result_df[metal].shift(3)
    
    return result_df

def update_dataset():
    """
    Główna funkcja do aktualizacji zbioru danych o nowe ceny metali.
    """
    # Ścieżka do pliku z danymi
    file_path = "data/processed/processed_data_raw.csv"
    
    # Sprawdź czy plik istnieje
    if not os.path.exists(file_path):
        print(f"Błąd: Plik {file_path} nie istnieje!")
        return
    
    # Wczytaj istniejące dane
    existing_data = pd.read_csv(file_path)
    print(f"Wczytano {len(existing_data)} wierszy z istniejącego zbioru danych.")
    
    # Znajdź ostatnią datę w zbiorze danych
    last_date = pd.to_datetime(existing_data['data'].max())
    start_date = (last_date + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    end_date = '2025-07-31'  # Do końca lipca 2025
    
    print(f"Pobieranie nowych danych od {start_date} do {end_date}...")
    
    # Pobierz nowe dane o cenach metali
    new_metal_data = get_metal_prices(start_date, end_date)
    
    if new_metal_data is None or new_metal_data.empty:
        print("Nie udało się pobrać nowych danych.")
        return
    
    print(f"Pobrano {len(new_metal_data)} nowych rekordów.")
    
    # Dodaj pozostałe kolumny z wartościami z ostatniego wiersza istniejących danych
    last_row = existing_data.iloc[-1].copy()
    
    # Pobierz nazwy kolumn ekonomicznych (bez metali i ich lag)
    econ_columns = [col for col in existing_data.columns 
                   if col not in ['data'] + 
                   ['Złoto', 'Srebro', 'Platyna', 'Pallad', 'Miedź'] +
                   [f'{metal}_lag_{lag}' for metal in ['Złoto', 'Srebro', 'Platyna', 'Pallad', 'Miedź'] for lag in [1, 3]]]
    
    # Dodaj kolumny ekonomiczne do nowych danych z wartościami z ostatniego wiersza
    for col in econ_columns:
        new_metal_data[col] = last_row[col]
    
    # Dodaj opóźnione wartości do całego zbioru danych
    # Najpierw połącz stare i nowe dane
    combined_data = pd.concat([existing_data, new_metal_data], ignore_index=True)
    combined_data['data'] = pd.to_datetime(combined_data['data'])
    combined_data = combined_data.sort_values('data')
    
    # Oblicz opóźnione wartości dla wszystkich danych
    for metal in ['Złoto', 'Srebro', 'Platyna', 'Pallad', 'Miedź']:
        combined_data[f'{metal}_lag_1'] = combined_data[metal].shift(1)
        combined_data[f'{metal}_lag_3'] = combined_data[metal].shift(3)
    
    # Usuń duplikaty dat jeśli istnieją
    combined_data = combined_data.drop_duplicates(subset=['data'])
    
    # Konwertuj datę z powrotem do formatu YYYY-MM-DD
    combined_data['data'] = combined_data['data'].dt.strftime('%Y-%m-%d')
    
    # Zapisz zaktualizowane dane do pliku
    combined_data.to_csv(file_path, index=False)
    print(f"Zaktualizowano zbiór danych. Nowy rozmiar: {len(combined_data)} wierszy.")
    print(f"Zakres dat: od {combined_data['data'].min()} do {combined_data['data'].max()}")

if __name__ == "__main__":
    update_dataset()
