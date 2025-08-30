# Skopiowano z model_training.py i zmodyfikowano dla backtestingu
# (zakomentowano plot='forecast')

import pandas as pd
import numpy as np
from pycaret.regression import *
from pathlib import Path
import os

# --- Konfiguracja --- 
# Używamy ścieżek względnych od głównego katalogu projektu
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / 'models'
RESULTS_BASE_DIR = PROJECT_DIR / 'results'
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_BASE_DIR.mkdir(exist_ok=True)

# Parametry dla setup PyCaret
SETUP_PARAMS = {
    'data': None, # Zostanie ustawione dynamicznie
    'target': None, # Zostanie ustawione dynamicznie
    'session_id': 42, # Dla powtarzalności
    'log_experiment': False, # Nie logujemy do MLflow w tym przykładzie
    'preprocess': True, # Włączamy preprocesing
    'imputation_type': 'simple', # Prosta imputacja brakujących wartości
    'numeric_imputation': 'mean', # Średnia dla numerycznych
    'categorical_imputation': 'mode', # Moda dla kategorycznych
    # --- ZMIANA: Wyłączamy usuwanie współliniowości na próbę --- 
    'remove_multicollinearity': False, # Usuń współliniowe cechy
    # ---------------------------------------------------------
    'multicollinearity_threshold': 0.9, # Próg współliniowości
    'normalize': True, # Normalizacja cech
    'normalize_method': 'zscore', # Metoda normalizacji
    'fold_strategy': 'timeseries', # Strategia walidacji krzyżowej dla szeregów czasowych
    # --- NOWOŚĆ: Wymagane dla 'timeseries' fold strategy --- 
    'data_split_shuffle': False,
    'fold_shuffle': False,
    # -----------------------------------------------------
    'fold': 3, # Liczba foldów
    'use_gpu': False, # Nie używamy GPU w tym przykładzie
    'n_jobs': -1 # Użyj wszystkich dostępnych rdzeni CPU
}

# --- Funkcja Trenująca --- 
def train_model(df: pd.DataFrame, target_column: str) -> bool:
    """
    Trenuje model regresyjny dla podanej kolumny docelowej i zapisuje go.
    Zwraca True jeśli trening i zapis się powiodły, False w przeciwnym razie.
    """
    print(f"\nRozpoczynanie treningu modelu dla: {target_column}")
    
    # Przygotowanie ścieżek specyficznych dla metalu
    model_name = f'model_{target_column}'
    results_dir = RESULTS_BASE_DIR / f'results_{target_column}'
    results_dir.mkdir(exist_ok=True)
    model_save_path = MODELS_DIR / model_name

    try:
        # Ustawienie parametrów dla konkretnego metalu
        setup_config = {
            'data': df, 
            'target': target_column,
            'train_size': 0.7, # PyCaret sam podzieli dane, ale my używamy danych już podzielonych w backtest.py
            'fold_strategy': 'timeseries', # Kluczowe dla szeregów czasowych
            'fold': 3, # Liczba foldów dla walidacji krzyżowej szeregów czasowych
            'numeric_imputation': 'mean', # <--- JAWNA IMPUTACJA
            'normalize': True, 
            'normalize_method': 'zscore',
            'session_id': 42,
            'verbose': False, # Zmniejszenie ilości logów
            'index': 'data', # <--- JAWNE USTAWIEIE INDEKSU CZASOWEGO
            'data_split_shuffle': False, # <--- WYMAGANE dla timeseries
            'fold_shuffle': False       # <--- WYMAGANE dla timeseries
        }

        # Usunięcie wierszy z brakującymi wartościami w kolumnie docelowej
        initial_rows = len(df)
        df.dropna(subset=[target_column], inplace=True)
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            print(f"[INFO] Usunięto {removed_rows} wierszy z brakującymi wartościami w kolumnie docelowej '{target_column}'.")

        # 1. Inicjalizacja środowiska PyCaret
        print("Inicjalizacja modelu...")
        # Przekaż rozpakowany słownik konfiguracji
        try:
            reg_setup = setup(**setup_config)
            print("Gotowe!")
        except Exception as e:
            print(f"Błąd podczas setup PyCaret: {e}")
            # Dodajmy więcej informacji o danych wejściowych
            print("Informacje o DataFrame przekazanym do setup:")
            df.info()
            print(df.head())
            return False # <--- Zwróć False w przypadku błędu

        # 2. Porównanie modeli
        print("\nWybór najlepszego modelu...")
        # Wykluczamy modele, które mogą być bardzo wolne lub niestabilne
        exclude_models = ['lar', 'par', 'ransac', 'tr'] 
        best_model = compare_models(exclude=exclude_models, sort='MAE') # Sortuj wg MAE, bo R2 może być ujemny
        print("Gotowe!")

        # 3. Dostrajanie najlepszego modelu (opcjonalne, ale często poprawia wyniki)
        print("\nDostrajanie modelu...")
        tuned_model = tune_model(best_model, optimize='MAE', n_iter=10) # n_iter - mała liczba dla szybszego działania
        print("Gotowe!")
        
        # 4. Finalizacja modelu (trening na całym zbiorze danych)
        print("\nFinalizacja modelu...")
        # Używamy dostrojonego modelu, jeśli jest lepszy, inaczej najlepszego
        final_model = finalize_model(tuned_model if tuned_model is not None else best_model)
        print("Gotowe!")

        # 5. Ewaluacja na zbiorze testowym (jeśli został utworzony przez setup)
        print("\nEwaluacja modelu...")
        evaluate_model(final_model)
        predictions = predict_model(final_model)
        print("Podsumowanie wyników:")
        # Drukujemy tylko metryki zwrócone przez predict_model, bo pull() może być niedostępne
        # lub zwracać metryki tylko z ostatniego folda, co jest mylące
        # Wyciągamy tabelę metryk (ostatnia tabela w predictions)
        if not predictions.empty:
            metrics_summary = predictions.iloc[-1:].filter(regex='MAE|MSE|RMSE|R2|RMSLE|MAPE')
            if not metrics_summary.empty:
                 print(metrics_summary)
            else:
                 print("Nie znaleziono tabeli metryk w wynikach predict_model.")

        # 6. Zapisywanie artefaktów
        print(f"\nZapisywanie modelu jako: {model_save_path}.pkl")
        save_model(final_model, str(model_save_path))
        
        # Zapisywanie wykresów analizy modelu
        print("Zapisywanie wykresów analizy...")
        # --- ZMIANA: Używamy save=True, aby pozwolić PyCaret na automatyczne zarządzanie ścieżką/nazwą --- 
        plot_model(final_model, plot='residuals', save=True)
        plot_model(final_model, plot='error', save=True)
        plot_model(final_model, plot='learning', save=True)
        plot_model(final_model, plot='feature', save=True)
        
        # --- ZAKOMENTOWANO DLA BACKTESTINGU --- 
        # Zakomentowano potencjalnie problematyczny wykres prognozy
        # try:
        #     # Próba wygenerowania wykresu prognozy, jeśli dane testowe są dostępne w setup
        #     test_indices = get_config('test_indices')
        #     if test_indices is not None and len(test_indices) > 0:
        #         plot_model(final_model, plot='forecast', data_kwargs={'fh': len(test_indices)}, save=str(results_dir / 'forecast_plot.png'))
        #     else:
        #         print("\n[INFO] Brak danych testowych w setup(), pomijanie wykresu 'forecast'.")
        # except Exception as plot_err:
        #     print(f"\n[WARNING] Nie udało się wygenerować wykresu 'forecast': {plot_err}")
        # ---------------------------------------
            
        # Wykres predykcji vs rzeczywiste wartości (używa hold-out set z setup)
        try:
            # --- ZMIANA: Używamy save=True --- 
            plot_model(final_model, plot='predictions', save=True)
        except Exception as e:
            print(f"Błąd podczas generowania wykresu predictions: {e}")

        # Zapisywanie metryk do pliku CSV
        try:
            # Pobierz metryki z ostatniego uruchomienia predict_model, jeśli dostępne
            if 'metrics_summary' in locals() and not metrics_summary.empty:
                 metrics_df = metrics_summary.reset_index(drop=True)
            else:
                 # Jeśli predict_model nie zwrócił metryk, spróbuj pobrać ze środowiska
                 metrics_df = pull()
                 # Sprawdź czy to DataFrame i czy zawiera oczekiwane kolumny
                 if not isinstance(metrics_df, pd.DataFrame) or not any(col in metrics_df.columns for col in ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']):
                      print("[WARNING] Nie udało się pobrać metryk za pomocą pull(). Zapis metryk pominięty.")
                      metrics_df = pd.DataFrame() # Pusty df, żeby reszta kodu działała
            
            if not metrics_df.empty:
                # Bierzemy tylko pierwszy wiersz, bo pull() może zwracać więcej wierszy dla CV
                metrics_to_save = metrics_df.head(1).filter(regex='Model|MAE|MSE|RMSE|R2|RMSLE|MAPE')
                metrics_file_path = results_dir / 'model_metrics.csv'
                metrics_to_save.to_csv(metrics_file_path, index=False)
                print(f"Metryki zapisano w: {metrics_file_path}")
                
        except Exception as e:
            print(f"Błąd podczas zapisywania metryk: {e}")
            
        print(f"\nGotowe! Model i wyniki zapisano w katalogu {results_dir}")
        return True # <--- Zwróć True w przypadku sukcesu

    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd podczas trenowania modelu dla {target_column}: {e}")
        import traceback
        traceback.print_exc()
        return False # <--- Zwróć False w przypadku innego błędu

# --- Funkcja główna (jeśli chcemy uruchomić ten plik bezpośrednio) ---
# (Pozostawiona dla spójności, ale nieużywana w kontekście backtest.py)
# def main():
#     # Wczytaj dane
#     print("Wczytywanie danych...")
#     # Domyślna ścieżka, jeśli uruchamiamy bezpośrednio
#     data_path = PROJECT_DIR / 'data' / 'raw' / 'combined_real_and_generated_data.csv'
#     df = load_combined_data(data_path) # Użyjemy load_combined_data z processing
#     if df is None:
#         print("Nie udało się załadować danych.")
#         return
    
#     # Wybierz metal do trenowania
#     metal_to_train = 'Złoto' # Przykładowo
    
#     # Trenuj model
#     train_model(df, metal_to_train)

# # Importuj funkcję ładowania danych z modułu processing
# try:
#      from src.processing import load_combined_data 
# except ImportError:
#      print("Nie można zaimportować 'load_combined_data' z src.processing. Upewnij się, że plik istnieje i jest poprawny.")
#      # Definiujmy prostą wersję zastępczą, żeby uniknąć błędu, jeśli import się nie powiedzie
#      def load_combined_data(filepath):
#           print("UWAGA: Używam zastępczej funkcji load_combined_data!")
#           try:
#                df = pd.read_csv(filepath)
#                df['data'] = pd.to_datetime(df['data'])
#                return df
#           except FileNotFoundError:
#                print(f"Nie znaleziono pliku: {filepath}")
#                return None
#           except Exception as e:
#                print(f"Błąd w zastępczej funkcji load_combined_data: {e}")
#                return None

# if __name__ == "__main__":
#     main()
