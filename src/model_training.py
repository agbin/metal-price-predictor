import pandas as pd
from pycaret.regression import *
import os
import logging
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)


def train_model(df: pd.DataFrame, metal: str, model_dir: str = 'models', results_dir: str = 'results') -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """
    Trenuje model regresyjny PyCaret dla określonego metalu.

    Args:
        df (pd.DataFrame): DataFrame zawierający przetworzone dane wejściowe 
                           (w tym kolumnę 'data' i obliczone lagi).
        metal (str): Nazwa kolumny metalu, która ma być prognozowana (np. 'Złoto').
        model_dir (str): Katalog do zapisania modelu.
        results_dir (str): Katalog do zapisania wyników (metryki, wykresy).

    Returns:
        Tuple[Optional[str], Optional[pd.DataFrame]]: Ścieżka do zapisanego modelu 
                                                    i DataFrame z metrykami, lub (None, None) w przypadku błędu.
    """
    logging.info(f"Rozpoczynanie treningu modelu dla: {metal}")

    if df is None or df.empty:
        logging.error("Dane wejściowe są puste lub None.")
        return None, None

    if metal not in df.columns:
        logging.error(f"Kolumna celu '{metal}' nie znaleziona w danych.")
        return None, None

    if 'data' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['data']):
        logging.error("Kolumna 'data' (datetime) nie znaleziona lub ma zły typ w danych wejściowych.")
        return None, None

    # Usuń wiersze, gdzie kolumna celu ma NaN - PyCaret tego wymaga
    initial_rows = len(df)
    df.dropna(subset=[metal], inplace=True)
    if initial_rows > len(df):
        logging.info(f"Usunięto {initial_rows - len(df)} wierszy z brakującymi wartościami w kolumnie docelowej '{metal}'.")

    if df.empty:
        logging.error(f"Po usunięciu NaN dla kolumny '{metal}' DataFrame jest pusty.")
        return None, None
        
    # Przygotuj ścieżki zapisu
    model_filename = f"model_{metal}"  # PyCaret automatycznie dodaje .pkl
    model_save_path = os.path.join(model_dir, model_filename)
    metal_results_dir = os.path.join(results_dir, f"results_{metal}")
    os.makedirs(metal_results_dir, exist_ok=True)
    metrics_save_path = os.path.join(metal_results_dir, "model_metrics.csv")
    
    try:
        # --- Konfiguracja PyCaret --- #
        logging.info("Inicjalizacja środowiska PyCaret...")
        # Używamy 'data' jako indeksu czasowego
        # session_id dla powtarzalności
        # numeric_features - PyCaret powinien sam wykryć, ale można podać jawnie
        # fold_strategy='timeseries', fold=3 dla walidacji czasowej
        # numeric_imputation='mean' lub 'knn' - obsługa ew. pozostałych NaN
        s = setup(data=df, 
                  target=metal, 
                  train_size=0.8, # Domyślny podział, jeśli nie robimy walidacji przedziałowej
                  data_split_shuffle=False, # WAŻNE: Bez tasowania dla szeregów czasowych
                  fold_strategy='timeseries', 
                  fold=3, # Mała liczba foldów dla mniejszych danych
                  fold_shuffle=False, # WAŻNE: Bez tasowania foldów
                  numeric_imputation='mean', # Imputacja pozostałych NaN (choć processing.py powinien je usunąć)
                  # UWAGA: PyCaret może mieć problem z kolumną 'data' jako feature.
                  # Można rozważyć jej usunięcie PRZED setup, jeśli nie jest potrzebna modelowi.
                  # LUB ustawić 'data' jako indeks PRZED setup: df.set_index('data', inplace=True)
                  # Spróbujmy na razie zostawić, PyCaret powinien ją zignorować jako feature
                  # jeśli jest typu datetime.
                  # W nowszych wersjach PyCaret można użyć index='data' argumentu
                  # index='data', # Można spróbować, jeśli wersja PyCaret > 3.0
                  ignore_features=['data'], # Jawne ignorowanie kolumny daty
                  session_id=123, 
                  verbose=True # Pokaż więcej informacji
                  )
        logging.info("Środowisko PyCaret skonfigurowane.")

        # --- Porównanie Modeli --- #
        logging.info("Porównywanie modeli...")
        # Wykluczamy modele, które mogą mieć problemy z małą ilością danych lub cechami
        # best_model = compare_models(exclude=['lar', 'par', 'ransac'])
        # Lub wybierzmy po prostu najlepszy:
        # Zmieniamy sortowanie, aby wybrać model z najlepszym (najniższym) MAPE
        best_model = compare_models(sort='MAPE')
        best_model_name = type(best_model).__name__

        # Zapisz nazwę najlepszego modelu do pliku
        model_name_filename = f"model_{metal}_name.txt"
        model_name_save_path = os.path.join(model_dir, model_name_filename)
        with open(model_name_save_path, 'w') as f:
            f.write(best_model_name)
        logging.info(f"Nazwa najlepszego modelu ('{best_model_name}') zapisana w: {model_name_save_path}")
        logging.info(f"Najlepszy znaleziony model: {type(best_model).__name__}")

        # --- Finalizacja Modelu --- #
        logging.info("Finalizowanie najlepszego modelu...")
        # Trenowanie na całym zbiorze danych (treningowym+walidacyjnym z setup)
        final_model = finalize_model(best_model)
        logging.info("Model sfinalizowany.")

        # --- Ewaluacja (opcjonalna, na zbiorze testowym z setup) --- #
        logging.info("Ewaluacja sfinalizowanego modelu...")
        # predict_model na hold-out set (jeśli train_size < 1)
        predictions = predict_model(final_model)
        # Metryki są zazwyczaj zwracane przez predict_model, ale można też je pobrać:
        metrics = pull()
        logging.info("Podsumowanie wyników ewaluacji:")
        print(metrics)
        # Zapisz metryki
        metrics.to_csv(metrics_save_path, index=False)
        logging.info(f"Metryki zapisano w: {metrics_save_path}")

        # --- Zapisywanie Modelu --- #
        logging.info(f"Zapisywanie modelu jako: {model_save_path}")
        save_model(final_model, model_save_path) # PyCaret zapisuje model i pipeline
        logging.info("Model zapisany pomyślnie.")

        # Usunięto generowanie wykresów, aby uprościć aplikację.

        return model_save_path, metrics

    except Exception as e:
        logging.error(f"Wystąpił błąd podczas trenowania modelu dla {metal}: {e}")
        import traceback
        traceback.print_exc() # Drukuj pełny traceback dla diagnostyki
        return None, None

# Przykład użycia (wymaga przetworzonych danych)
if __name__ == '__main__':
    from src.processing import load_combined_data
    processed_data = load_combined_data() # Używamy nowej funkcji z lagami
    if processed_data is not None:
        model_path, model_metrics = train_model(processed_data, 'Złoto')
        if model_path:
            print(f"Trening dla Złota zakończony. Model zapisany w: {model_path}")
            print("Metryki modelu:")
            print(model_metrics)
        else:
            print("Trening dla Złota nie powiódł się.")
