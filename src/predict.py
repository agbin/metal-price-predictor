import pandas as pd
from pycaret.regression import load_model, predict_model
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)

def predict_price(future_data: pd.DataFrame, metal: str) -> Optional[float]:
    """
    Generuje prognozę ceny dla podanego metalu na podstawie przygotowanego DataFrame.

    Args:
        future_data (pd.DataFrame): DataFrame zawierający jedną linię z danymi 
                                   wejściowymi dla modelu, w tym obliczone lagi.
        metal (str): Nazwa metalu (np. 'Złoto', 'Srebro'), używana do załadowania 
                     odpowiedniego modelu.

    Returns:
        float: Prognozowana cena lub None w przypadku błędu.
    """
    try:
        # Sprawdzenie czy future_data to DataFrame z jedną linią
        if not isinstance(future_data, pd.DataFrame) or len(future_data) != 1:
            logging.error("predict_price: Oczekiwano DataFrame z jedną linią.")
            return None

        model_filename = f"model_{metal}"  # PyCaret automatycznie dodaje .pkl
        model_path = os.path.join('models', model_filename)
        model_path_with_pkl = model_path + '.pkl'  # Dla sprawdzenia istnienia pliku

        if not os.path.exists(model_path_with_pkl):
            logging.error(f"Nie znaleziono zapisanego modelu dla {metal} w {model_path_with_pkl}")
            return None

        # Załadowanie zapisanego modelu (PyCaret load_model automatycznie dodaje .pkl)
        saved_model = load_model(model_path)
        logging.info(f"Załadowano model dla {metal} z {model_path}")

        # Usunięcie kolumny celu z danych wejściowych, jeśli istnieje 
        # (predict_model tego wymaga)
        if metal in future_data.columns:
             input_data = future_data.drop(columns=[metal])
        else:
             input_data = future_data
        
        # Usunięcie kolumny 'data' jeśli istnieje, bo nie jest cechą modelu
        if 'data' in input_data.columns:
            input_data = input_data.drop(columns=['data'])

        logging.info(f"Dane wejściowe do predict_model (kolumny): {input_data.columns.tolist()}")
        logging.info(f"Dane wejściowe do predict_model (kształt): {input_data.shape}")

        # Generowanie prognozy
        predictions = predict_model(saved_model, data=input_data)

        # Sprawdzenie czy wynik zawiera kolumnę 'prediction_label'
        if 'prediction_label' in predictions.columns:
            predicted_value = predictions['prediction_label'].iloc[0]
            logging.info(f"Prognoza dla {metal}: {predicted_value}")
            return predicted_value
        else:
            logging.error("Kolumna 'prediction_label' nie znaleziona w wyniku predict_model.")
            print("Wynik predict_model:", predictions)
            return None

    except Exception as e:
        logging.error(f"Błąd podczas prognozowania ceny dla {metal}: {e}")
        print(f"Błąd podczas prognozowania ceny dla {metal}: {e}") # Dodatkowy print dla widoczności w Streamlit
        # Zwróć uwagę na szczegóły błędu w logach
        import traceback
        traceback.print_exc()
        return None

# Przykład użycia (wymaga istnienia modelu i odpowiednich danych)
# if __name__ == '__main__':
#     # Przykładowe dane wejściowe (muszą zawierać WSZYSTKIE kolumny oczekiwane przez model)
#     # Wartości są tylko przykładem!
#     example_data = pd.DataFrame({
#         'data': [pd.to_datetime('2025-02-01')],
#         'Srebro': [23.5], 'Platyna': [950], 'Pallad': [1000], 'Miedź': [4.0],
#         'inflacja': [2.5], 'stopy_procentowe': [1.75], 'bezrobocie': [3.8],
#         'pkb': [1.5], 'pkb_global': [2.0], 'kurs_usd': [4.0], 'indeks_vix': [15.0],
#         'Złoto_lag_1': [2050],  # Rzeczywista cena złota z poprzedniego miesiąca
#         'Złoto_lag_3': [2000],  # Rzeczywista cena złota sprzed 3 miesięcy
#         'Srebro_lag_1': [23.0], 'Srebro_lag_3': [22.0], # Itd. dla innych metali
#         'Platyna_lag_1': [940], 'Platyna_lag_3': [900],
#         'Pallad_lag_1': [980], 'Pallad_lag_3': [950],
#         'Miedź_lag_1': [3.9], 'Miedź_lag_3': [3.8]
#         # Potencjalnie brakuje tu jeszcze kolumny Złoto, która zostanie usunięta przed predykcją
#     }, index=[0])
# 
#     prognoza_zlota = predict_price(example_data.copy(), 'Złoto') # Użyj kopii, bo funkcja może modyfikować df
#     if prognoza_zlota is not None:
#         print(f"Przykładowa prognozowana cena złota: {prognoza_zlota:.2f}")
#     else:
#         print("Nie udało się uzyskać prognozy złota.")
