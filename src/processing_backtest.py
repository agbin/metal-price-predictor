import pandas as pd

def load_data_for_backtest(file_path: str):
    """Ładuje dane z pliku CSV specjalnie na potrzeby backtestu.

    Args:
        file_path (str): Ścieżka do pliku CSV.

    Returns:
        pd.DataFrame or None: DataFrame z danymi (z kolumną 'data' sparsowaną jako datetime) 
                             lub None, jeśli plik nie istnieje.
    """
    try:
        # Ładuj dane, od razu parsując kolumnę 'data'
        df = pd.read_csv(file_path, parse_dates=['data'])
        print(f"Załadowano dane z {file_path}. Kolumna 'data' sparsowana.")
        
        # Sortowanie po dacie
        df = df.sort_values(by='data').reset_index(drop=True)
        print(f"Okres danych: od {df['data'].min()} do {df['data'].max()}")
        return df
    except FileNotFoundError:
        print(f"BŁĄD: Plik danych '{file_path}' nie został znaleziony.")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd podczas ładowania danych: {e}")
        return None
