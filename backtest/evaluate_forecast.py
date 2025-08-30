"""
Skrypt do oceny skuteczności prognoz modelu poprzez:
1. Trenowanie na danych do końca 2023
2. Prognozowanie wartości na 2024-07.2025
3. Porównanie z rzeczywistymi wartościami

Nie modyfikuje głównego kodu aplikacji.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from datetime import datetime

# Dodaj ścieżkę głównego projektu do sys.path
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent
sys.path.append(str(project_dir))

# Importy z głównej aplikacji
from src.processing import load_combined_data
from src.model_training import train_model
from pycaret.regression import load_model, predict_model

# Konfiguracja katalogów
BACKTEST_DIR = current_dir
BACKTEST_MODELS_DIR = BACKTEST_DIR / "models"
BACKTEST_RESULTS_DIR = BACKTEST_DIR / "results"
BACKTEST_PLOTS_DIR = BACKTEST_DIR / "plots"

# Utwórz katalogi jeśli nie istnieją
os.makedirs(BACKTEST_MODELS_DIR, exist_ok=True)
os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)
os.makedirs(BACKTEST_PLOTS_DIR, exist_ok=True)

def load_and_split_data(end_training_date='2023-12-31'):
    """Ładuje dane i dzieli je na zbiór treningowy (do end_training_date) i testowy (po end_training_date)"""
    print(f"Ładowanie danych...")
    df = load_combined_data()
    
    if df is None or df.empty:
        print("Błąd: Nie można załadować danych!")
        return None, None
        
    if 'data' not in df.columns:
        print("Błąd: Brak kolumny 'data' w danych!")
        return None, None
        
    # Upewnij się, że 'data' jest w formacie datetime
    df['data'] = pd.to_datetime(df['data'])
    
    # Podziel dane na treningowe i testowe
    train_data = df[df['data'] <= end_training_date].copy()
    test_data = df[df['data'] > end_training_date].copy()
    
    print(f"Dane treningowe: {len(train_data)} wierszy od {train_data['data'].min()} do {train_data['data'].max()}")
    print(f"Dane testowe: {len(test_data)} wierszy od {test_data['data'].min()} do {test_data['data'].max()}")
    
    return train_data, test_data

def train_backtest_model(train_data, metal):
    """Trenuje model dla backtestingu"""
    print(f"Trenowanie modelu dla {metal}...")
    model_path, metrics = train_model(
        train_data, 
        metal, 
        model_dir=str(BACKTEST_MODELS_DIR),
        results_dir=str(BACKTEST_RESULTS_DIR)
    )
    return model_path, metrics

def recursive_forecast(initial_data, model_path, metal, months_ahead):
    """
    Generuje prognozy rekurencyjnie na określoną liczbę miesięcy do przodu.
    Dla każdej kolejnej prognozy aktualizuje wartości opóźnione (lagi).
    """
    print(f"Generowanie prognoz rekurencyjnie na {months_ahead} miesięcy dla {metal}...")
    
    # Załaduj model
    model = load_model(model_path)
    
    # Przygotuj dane początkowe - ostatni wiersz danych treningowych
    forecast_row = initial_data.iloc[-1:].copy()
    
    # Lista dostępnych metali
    available_metals = ["Złoto", "Srebro", "Platyna", "Pallad", "Miedź"]
    
    # Przechowuj historię prognoz dla każdego metalu
    forecasts = {}
    for m in available_metals:
        forecasts[m] = []
    
    # Przechowuj daty prognoz
    forecast_dates = []
    
    # Generuj prognozy
    last_date = pd.to_datetime(forecast_row['data'].values[0])
    
    for i in range(months_ahead):
        # Oblicz nową datę (miesiąc do przodu)
        next_date = last_date + pd.DateOffset(months=i+1)
        forecast_dates.append(next_date)
        
        # Aktualizuj datę w wierszu prognozującym
        forecast_row['data'] = next_date
        
        # Aktualizuj wartości opóźnione na podstawie poprzednich prognoz
        if i > 0:
            # Aktualizuj lag_1 dla wszystkich metali
            for m in available_metals:
                if len(forecasts[m]) >= 1:
                    forecast_row[f'{m}_lag_1'] = forecasts[m][-1]
                
                # Aktualizuj lag_3 jeśli mamy już 3 prognozy
                if len(forecasts[m]) >= 3:
                    forecast_row[f'{m}_lag_3'] = forecasts[m][-3]
        
        # Usuń kolumnę celu (jeśli istnieje) i kolumnę daty przed predykcją
        input_data = forecast_row.copy()
        if metal in input_data.columns:
            input_data = input_data.drop(columns=[metal])
        if 'data' in input_data.columns:
            input_data = input_data.drop(columns=['data'])
        
        # Wykonaj predykcję
        predictions = predict_model(model, data=input_data)
        prediction = predictions['prediction_label'].iloc[0]
        
        # Zapisz prognozę
        forecasts[metal].append(prediction)
        
        # Aktualizuj wartość w wierszu prognozującym dla kolejnych iteracji
        forecast_row[metal] = prediction
    
    # Utwórz DataFrame z wynikami dla głównego metalu
    results_df = pd.DataFrame({
        'data': forecast_dates,
        'prognoza': forecasts[metal]
    })
    
    return results_df

def compare_with_actual(forecasts_df, actual_df, metal):
    """Porównuje prognozy z rzeczywistymi wartościami i oblicza metryki błędów"""
    print(f"Porównywanie prognoz z rzeczywistymi wartościami dla {metal}...")
    
    # Upewnij się, że obie ramki danych mają datę jako indeks
    # Najpierw sprawdź, czy kolumny istnieją
    print(f"Kolumny w forecasts_df: {forecasts_df.columns.tolist()}")
    print(f"Kolumny w actual_df: {actual_df.columns.tolist()}")
    
    # Konwertuj forecasts_df
    forecasts_df = forecasts_df.copy()
    forecasts_df['data'] = pd.to_datetime(forecasts_df['data'])
    forecasts_df.set_index('data', inplace=True)
    
    # Konwertuj actual_df - upewnij się, że używamy właściwej kolumny daty
    actual_df = actual_df.copy()
    # Sprawdź, czy kolumna 'data' istnieje
    date_column = None
    for col in actual_df.columns:
        if col == 'data' or 'date' in col.lower() or 'data' in col.lower():
            date_column = col
            break
    
    if date_column is None:
        print("Błąd: Nie znaleziono kolumny daty w rzeczywistych danych!")
        print(f"Dostępne kolumny: {actual_df.columns.tolist()}")
        return None
    
    actual_df['data'] = pd.to_datetime(actual_df[date_column])
    actual_df.set_index('data', inplace=True)
    
    # Wybierz tylko wspólne daty
    common_dates = forecasts_df.index.intersection(actual_df.index)
    if len(common_dates) == 0:
        print("Brak wspólnych dat do porównania!")
        return None
        
    print(f"Liczba punktów do porównania: {len(common_dates)}")
    
    # Wybierz dane do porównania
    forecasts_subset = forecasts_df.loc[common_dates, 'prognoza']
    actual_subset = actual_df.loc[common_dates, metal]
    
    # Oblicz metryki błędów
    mape = mean_absolute_percentage_error(actual_subset, forecasts_subset) * 100
    rmse = np.sqrt(mean_squared_error(actual_subset, forecasts_subset))
    r2 = r2_score(actual_subset, forecasts_subset)
    
    print(f"Metryki dla {metal}:")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Stwórz DataFrame z wynikami do porównania
    comparison_df = pd.DataFrame({
        'rzeczywiste': actual_subset,
        'prognozowane': forecasts_subset,
        'błąd_procentowy': abs((forecasts_subset - actual_subset) / actual_subset) * 100
    })
    
    return comparison_df, mape, rmse, r2

def plot_comparison(comparison_df, metal, save_path):
    """Generuje wykres porównujący rzeczywiste wartości z prognozowanymi"""
    print(f"Generowanie wykresu dla {metal}...")
    
    plt.figure(figsize=(12, 6))
    
    # Wykres wartości rzeczywistych
    plt.plot(comparison_df.index, comparison_df['rzeczywiste'], 'b-', label='Rzeczywiste')
    
    # Wykres wartości prognozowanych
    plt.plot(comparison_df.index, comparison_df['prognozowane'], 'r--', label='Prognozowane')
    
    # Dodaj tytuł i etykiety
    plt.title(f'Porównanie wartości rzeczywistych i prognozowanych - {metal}')
    plt.xlabel('Data')
    plt.ylabel('Cena')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Zapisz wykres
    plt.savefig(save_path)
    plt.close()

def main():
    """Główna funkcja wykonująca backtest dla wszystkich metali"""
    # Lista metali do prognozowania
    metals = ["Złoto", "Srebro", "Platyna", "Pallad", "Miedź"]
    end_training_date = '2023-12-31'
    
    # Załaduj i podziel dane
    train_data, test_data = load_and_split_data(end_training_date)
    if train_data is None or test_data is None:
        return
    
    # Zapisz wyniki do pliku
    results_file = BACKTEST_RESULTS_DIR / "backtest_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Wyniki backtestingu - {datetime.now()}\n")
        f.write(f"Dane treningowe do: {end_training_date}\n")
        f.write("-" * 50 + "\n\n")
    
    # Dla każdego metalu
    all_results = []
    for metal in metals:
        # Trenuj model
        model_path, _ = train_backtest_model(train_data, metal)
        if not model_path:
            print(f"Błąd: Nie udało się wytrenować modelu dla {metal}")
            continue
            
        # Generuj prognozy
        months_ahead = len(test_data)
        forecast_df = recursive_forecast(train_data, model_path, metal, months_ahead)
        
        # Porównaj z rzeczywistymi wartościami
        comparison_result = compare_with_actual(forecast_df, test_data, metal)
        if comparison_result:
            comparison_df, mape, rmse, r2 = comparison_result
            
            # Zapisz wyniki do pliku CSV
            comparison_df.to_csv(BACKTEST_RESULTS_DIR / f"comparison_{metal}.csv")
            
            # Generuj wykres
            plot_path = BACKTEST_PLOTS_DIR / f"comparison_{metal}.png"
            plot_comparison(comparison_df, metal, plot_path)
            
            # Zapisz metryki do głównego pliku wyników
            with open(results_file, 'a') as f:
                f.write(f"Metryki dla {metal}:\n")
                f.write(f"MAPE: {mape:.2f}%\n")
                f.write(f"RMSE: {rmse:.2f}\n")
                f.write(f"R²: {r2:.4f}\n")
                f.write("-" * 30 + "\n\n")
            
            # Dodaj do zbiorczych wyników
            all_results.append({
                'metal': metal,
                'mape': mape,
                'rmse': rmse,
                'r2': r2
            })
    
    # Stwórz zbiorczy DataFrame z metrykami
    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\nPodsumowanie wyników backtestingu:")
        print(results_df)
        results_df.to_csv(BACKTEST_RESULTS_DIR / "metrics_summary.csv", index=False)
        
        # Zapisz podsumowanie do pliku wyników
        with open(results_file, 'a') as f:
            f.write("\nPodsumowanie wyników:\n")
            f.write(results_df.to_string())
    
    print(f"\nBacktesting zakończony. Wyniki zapisano w {BACKTEST_RESULTS_DIR}")

if __name__ == "__main__":
    main()
