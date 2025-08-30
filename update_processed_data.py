#!/usr/bin/env python
"""
Skrypt aktualizujący processed_data.csv na podstawie combined_real_and_generated_data.csv
do lipca 2025, z wykorzystaniem pełnego przetwarzania danych.
"""

import os
import pandas as pd
from datetime import datetime
import sys

# Dodaj katalog src do ścieżki, aby móc importować moduły
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.processing import process_pipeline

def update_processed_data():
    """
    Aktualizuje plik processed_data.csv do lipca 2025
    używając pełnego pipeline'a przetwarzania danych.
    """
    # Ścieżki do plików
    input_filepath = os.path.join("data", "processed", "combined_real_and_generated_data.csv")
    output_filepath = os.path.join("data", "processed", "processed_data.csv")
    
    # Sprawdź czy pliki wejściowe istnieją
    if not os.path.exists(input_filepath):
        print(f"Błąd: Plik wejściowy {input_filepath} nie istnieje!")
        return
        
    # Sprawdź aktualny zakres dat w pliku wyjściowym (jeśli istnieje)
    if os.path.exists(output_filepath):
        existing_data = pd.read_csv(output_filepath)
        if 'data' in existing_data.columns:
            existing_data['data'] = pd.to_datetime(existing_data['data'])
            min_date = existing_data['data'].min()
            max_date = existing_data['data'].max()
            print(f"Istniejący zakres dat w {output_filepath}: od {min_date.strftime('%Y-%m-%d')} do {max_date.strftime('%Y-%m-%d')}")
            print(f"Liczba wierszy: {len(existing_data)}")
    
    # Uruchom pełny pipeline przetwarzania danych
    print(f"Uruchamianie pełnego pipeline'a przetwarzania danych...")
    process_pipeline(input_filepath, output_filepath)
    
    # Sprawdź zaktualizowany plik
    if os.path.exists(output_filepath):
        updated_data = pd.read_csv(output_filepath)
        if 'data' in updated_data.columns:
            updated_data['data'] = pd.to_datetime(updated_data['data'])
            min_date = updated_data['data'].min()
            max_date = updated_data['data'].max()
            print(f"Zaktualizowany zakres dat w {output_filepath}: od {min_date.strftime('%Y-%m-%d')} do {max_date.strftime('%Y-%m-%d')}")
            print(f"Liczba wierszy: {len(updated_data)}")
            print(f"Plik {output_filepath} został pomyślnie zaktualizowany!")

if __name__ == "__main__":
    update_processed_data()
