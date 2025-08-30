import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os
from src.predict import predict_price
from dateutil.relativedelta import relativedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_actual_prices(symbol='GC=F', start_date='2024-01-01', end_date='2025-08-01'):
    """
    Pobiera rzeczywiste ceny złota z Yahoo Finance.
    
    Args:
        symbol (str): Symbol giełdowy złota
        start_date (str): Data początkowa w formacie 'YYYY-MM-DD'
        end_date (str): Data końcowa w formacie 'YYYY-MM-DD'
        
    Returns:
        pd.DataFrame: DataFrame zawierający daty i ceny zamknięcia
    """
    try:
        logging.info(f"Pobieranie danych dla {symbol} od {start_date} do {end_date}")
        data = yf.download(symbol, start=start_date, end=end_date)
        
        if data.empty:
            logging.error(f"Nie udało się pobrać danych dla {symbol}")
            return None
            
        # Przygotowujemy dane - wybieramy tylko ceny zamknięcia i resetujemy indeks
        result = data[['Close']].reset_index()
        result.columns = ['data', 'cena']
        
        # Konwertujemy na dane miesięczne (bierzemy ostatni dzień miesiąca)
        result['month_year'] = result['data'].dt.to_period('M')
        monthly_data = result.groupby('month_year').last().reset_index()
        monthly_data['data'] = monthly_data['month_year'].dt.to_timestamp()
        monthly_data = monthly_data[['data', 'cena']]
        
        logging.info(f"Pobrano {len(monthly_data)} miesięcznych rekordów dla {symbol}")
        return monthly_data
        
    except Exception as e:
        logging.error(f"Wystąpił błąd podczas pobierania danych: {e}")
        return None

def prepare_forecast_input(latest_data, forecast_dates):
    """
    Przygotowuje dane wejściowe do prognozowania dla podanych dat.
    
    Args:
        latest_data (pd.DataFrame): DataFrame z ostatnimi dostępnymi danymi
        forecast_dates (list): Lista dat, dla których chcemy generować prognozy
        
    Returns:
        dict: Słownik, gdzie kluczami są daty prognoz, a wartościami DataFrames z danymi wejściowymi
    """
    forecast_inputs = {}
    
    # Zakładamy, że latest_data ma już wszystkie potrzebne kolumny
    # Sprawdzamy, czy mamy kolumny z opóźnionymi wartościami dla złota
    required_lag_cols = ['Złoto_lag_1', 'Złoto_lag_3']
    if not all(col in latest_data.columns for col in required_lag_cols):
        logging.error(f"Brakuje wymaganych kolumn w danych: {required_lag_cols}")
        return None
    
    # Przygotowujemy dane wejściowe dla każdej daty prognozy
    for forecast_date in forecast_dates:
        # Kopiujemy ostatnie dostępne dane jako punkt wyjścia
        input_data = latest_data.copy()
        
        # Aktualizujemy datę
        input_data['data'] = pd.to_datetime(forecast_date)
        
        # Zapisujemy dane wejściowe dla tej daty
        forecast_inputs[forecast_date] = input_data
    
    return forecast_inputs

def generate_forecasts(input_data_dict, metal='Złoto'):
    """
    Generuje prognozy dla podanych danych wejściowych.
    
    Args:
        input_data_dict (dict): Słownik z danymi wejściowymi (data -> DataFrame)
        metal (str): Nazwa metalu
        
    Returns:
        pd.DataFrame: DataFrame z datami i prognozami
    """
    results = []
    
    for date, input_data in input_data_dict.items():
        # Generujemy prognozę
        predicted_value = predict_price(input_data, metal)
        
        if predicted_value is not None:
            results.append({
                'data': date,
                'prognoza': predicted_value
            })
        else:
            logging.error(f"Nie udało się wygenerować prognozy dla daty {date}")
    
    if results:
        return pd.DataFrame(results)
    else:
        return None

def plot_comparison(actual_data, forecast_data):
    """
    Tworzy wykres porównujący rzeczywiste ceny z prognozami.
    
    Args:
        actual_data (pd.DataFrame): DataFrame z rzeczywistymi cenami
        forecast_data (pd.DataFrame): DataFrame z prognozami
    """
    plt.figure(figsize=(12, 6))
    
    # Rysujemy rzeczywiste ceny
    plt.plot(actual_data['data'], actual_data['cena'], 'b-', label='Rzeczywiste ceny')
    
    # Rysujemy prognozy
    plt.plot(forecast_data['data'], forecast_data['prognoza'], 'r--', label='Prognozy modelu')
    
    plt.title('Porównanie rzeczywistych cen złota z prognozami modelu (2024-2025)')
    plt.xlabel('Data')
    plt.ylabel('Cena (USD)')
    plt.legend()
    plt.grid(True)
    
    # Zapisujemy wykres
    output_dir = 'results/forecasts'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/gold_forecast_comparison.png')
    
    plt.show()

def calculate_metrics(actual_data, forecast_data):
    """
    Oblicza metryki porównujące rzeczywiste ceny z prognozami.
    
    Args:
        actual_data (pd.DataFrame): DataFrame z rzeczywistymi cenami
        forecast_data (pd.DataFrame): DataFrame z prognozami
        
    Returns:
        dict: Słownik z metrykami
    """
    # Łączymy dane
    merged_data = pd.merge(actual_data, forecast_data, on='data', how='inner')
    
    if merged_data.empty:
        logging.error("Brak wspólnych dat do porównania")
        return None
    
    # Obliczamy błędy
    merged_data['error'] = merged_data['cena'] - merged_data['prognoza']
    merged_data['abs_error'] = np.abs(merged_data['error'])
    merged_data['squared_error'] = merged_data['error'] ** 2
    merged_data['percentage_error'] = (merged_data['abs_error'] / merged_data['cena']) * 100
    
    # Obliczamy metryki
    mae = merged_data['abs_error'].mean()
    rmse = np.sqrt(merged_data['squared_error'].mean())
    mape = merged_data['percentage_error'].mean()
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'num_points': len(merged_data)
    }

def load_latest_processed_data():
    """
    Ładuje ostatnie przetworzone dane, które będą służyć jako podstawa do prognoz.
    
    Returns:
        pd.DataFrame: DataFrame z przetworzonymi danymi
    """
    processed_data_path = os.path.join('data', 'processed', 'processed_data.csv')
    
    if not os.path.exists(processed_data_path):
        logging.error(f"Plik z przetworzonymi danymi nie istnieje: {processed_data_path}")
        return None
    
    try:
        df = pd.read_csv(processed_data_path)
        df['data'] = pd.to_datetime(df['data'])
        
        # Sortujemy dane według daty i bierzemy ostatni wiersz
        df = df.sort_values('data').reset_index(drop=True)
        latest_data = df.iloc[[-1]].copy()
        
        logging.info(f"Załadowano ostatnie przetworzone dane z {processed_data_path}")
        logging.info(f"Data ostatnich danych: {latest_data['data'].iloc[0]}")
        
        return latest_data
        
    except Exception as e:
        logging.error(f"Wystąpił błąd podczas ładowania przetworzonych danych: {e}")
        return None

def update_lag_features(input_data, actual_prices, metal='Złoto'):
    """
    Aktualizuje cechy opóźnione (lagi) na podstawie rzeczywistych cen.
    
    Args:
        input_data (pd.DataFrame): DataFrame z danymi wejściowymi
        actual_prices (pd.DataFrame): DataFrame z rzeczywistymi cenami
        metal (str): Nazwa metalu
        
    Returns:
        dict: Słownik z aktualizowanymi danymi wejściowymi dla każdej daty
    """
    updated_inputs = {}
    
    # Sortujemy rzeczywiste ceny według daty
    actual_prices = actual_prices.sort_values('data').reset_index(drop=True)
    
    # Dla każdego miesiąca, przygotowujemy dane wejściowe
    for i in range(len(actual_prices)):
        current_date = actual_prices['data'].iloc[i]
        
        # Kopiujemy dane wejściowe
        current_input = input_data.copy()
        
        # Aktualizujemy datę
        current_input['data'] = current_date
        
        # Aktualizujemy lag_1 (cena z poprzedniego miesiąca)
        if i > 0:
            current_input[f'{metal}_lag_1'] = actual_prices['cena'].iloc[i-1]
        
        # Aktualizujemy lag_3 (cena sprzed 3 miesięcy)
        if i > 2:
            current_input[f'{metal}_lag_3'] = actual_prices['cena'].iloc[i-3]
        elif i > 0:  # Jeśli nie mamy danych sprzed 3 miesięcy, używamy najstarszych dostępnych
            current_input[f'{metal}_lag_3'] = actual_prices['cena'].iloc[0]
        
        # Zapisujemy aktualizowane dane wejściowe
        if i > 2:  # Zapisujemy tylko jeśli mamy wystarczająco danych dla lag_3
            updated_inputs[current_date] = current_input
    
    return updated_inputs

def main():
    """
    Główna funkcja wykonująca porównanie prognoz z rzeczywistymi cenami.
    """
    # Definiujemy okres do porównania
    start_date = '2024-01-01'
    end_date = '2025-08-01'
    
    # Pobieramy rzeczywiste ceny
    actual_prices = get_actual_prices(start_date=start_date, end_date=end_date)
    if actual_prices is None:
        logging.error("Nie udało się pobrać rzeczywistych cen. Kończenie programu.")
        return
    
    # Ładujemy ostatnie przetworzone dane jako podstawę do prognoz
    latest_data = load_latest_processed_data()
    if latest_data is None:
        logging.error("Nie udało się załadować danych wejściowych. Kończenie programu.")
        return
    
    # Aktualizujemy cechy opóźnione na podstawie rzeczywistych cen
    input_data_dict = update_lag_features(latest_data, actual_prices)
    if not input_data_dict:
        logging.error("Nie udało się przygotować danych wejściowych. Kończenie programu.")
        return
    
    # Generujemy prognozy
    forecast_data = generate_forecasts(input_data_dict)
    if forecast_data is None:
        logging.error("Nie udało się wygenerować prognoz. Kończenie programu.")
        return
    
    # Obliczamy metryki
    metrics = calculate_metrics(actual_prices, forecast_data)
    if metrics is None:
        logging.error("Nie udało się obliczyć metryk. Kończenie programu.")
        return
    
    # Wyświetlamy metryki
    print("\n=== Metryki porównania prognoz z rzeczywistymi cenami ===")
    print(f"Liczba punktów do porównania: {metrics['num_points']}")
    print(f"MAE (Mean Absolute Error): {metrics['MAE']:.2f} USD")
    print(f"RMSE (Root Mean Square Error): {metrics['RMSE']:.2f} USD")
    print(f"MAPE (Mean Absolute Percentage Error): {metrics['MAPE']:.2f}%")
    
    # Rysujemy wykres
    plot_comparison(actual_prices, forecast_data)
    
    # Zapisujemy wyniki do CSV
    output_dir = 'results/forecasts'
    os.makedirs(output_dir, exist_ok=True)
    
    merged_data = pd.merge(actual_prices, forecast_data, on='data', how='inner')
    merged_data.columns = ['data', 'rzeczywista_cena', 'prognoza']
    merged_data['błąd_bezwzględny'] = np.abs(merged_data['rzeczywista_cena'] - merged_data['prognoza'])
    merged_data['błąd_procentowy'] = (merged_data['błąd_bezwzględny'] / merged_data['rzeczywista_cena']) * 100
    
    merged_data.to_csv(f'{output_dir}/gold_forecast_comparison.csv', index=False)
    print(f"\nWyniki zapisano w pliku: {output_dir}/gold_forecast_comparison.csv")
    print(f"Wykres zapisano w pliku: {output_dir}/gold_forecast_comparison.png")

if __name__ == "__main__":
    main()
