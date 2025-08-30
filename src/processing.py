from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import logging

logging.basicConfig(level=logging.INFO)

# Lista potencjalnych kolumn z cenami metali
METAL_COLUMNS = ['Złoto', 'Srebro', 'Platyna', 'Pallad', 'Miedź']

def load_combined_data(filepath=None) -> pd.DataFrame:
    """
    Ładuje dane z pliku CSV, konwertuje datę, dodaje lagi dla metali i usuwa NaN.
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "data", "processed", "combined_real_and_generated_data.csv")
    
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Wczytano dane z {filepath}")

        # Sprawdzenie i konwersja kolumny 'data'
        if 'data' in df.columns:
            df['data'] = pd.to_datetime(df['data'])
            logging.info("Kolumna 'data' przekonwertowana na datetime.")
            # Sortowanie wg daty - kluczowe dla lagów!
            df = df.sort_values('data').reset_index(drop=True)
        else:
            logging.error("Brak kolumny 'data' w pliku.")
            return None

        # --- NOWOŚĆ: Dodawanie Lagów --- #
        logging.info("Dodawanie cech opóźnionych (lag_1, lag_3) dla metali...")
        metals_present = [metal for metal in METAL_COLUMNS if metal in df.columns]
        lag_columns = []

        for metal in metals_present:
            # Sprawdzenie czy kolumna metalu zawiera NaN PRZED lagowaniem
            if df[metal].isnull().any():
                logging.warning(f"Kolumna '{metal}' zawiera NaN PRZED lagowaniem. Rozważ imputację lub usunięcie.")
                # Na razie kontynuujemy, PyCaret powinien to obsłużyć lub usuniemy później

            lag1_col = f'{metal}_lag_1'
            lag3_col = f'{metal}_lag_3'
            df[lag1_col] = df[metal].shift(1)
            df[lag3_col] = df[metal].shift(3)
            lag_columns.extend([lag1_col, lag3_col])
            logging.info(f"Dodano lagi dla: {metal}")

        # --- Usuwanie NaN --- # 
        # Usuwamy wiersze, gdzie jakikolwiek lag LUB jakakolwiek oryginalna cena metalu ma NaN
        # To zapewnia, że mamy kompletne dane dla co najmniej jednego metalu w każdym wierszu
        # oraz że nie mamy NaN w lagach, które mogą powstać na początku.
        initial_rows = len(df)
        # Lista kolumn do sprawdzenia pod kątem NaN (oryginalne metale + wszystkie lagi)
        cols_to_check_for_nan = metals_present + lag_columns
        df.dropna(subset=cols_to_check_for_nan, how='any', inplace=True)
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logging.info(f"Usunięto {removed_rows} początkowych wierszy z powodu NaN w lagach lub cenach metali.")
        else:
            logging.info("Nie usunięto wierszy z powodu NaN w lagach/cenach (prawdopodobnie brak NaN).")
        # --------------------------------

        logging.info(f"Zwracanie przetworzonych danych. Liczba wierszy: {len(df)}, Liczba kolumn: {len(df.columns)}")
        return df

    except FileNotFoundError:
        logging.error(f"Błąd: Plik {filepath} nie został znaleziony.")
        return None
    except Exception as e:
        logging.error(f"Wystąpił nieoczekiwany błąd podczas ładowania danych: {e}")
        return None

def fill_gdp_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uzupełnia brakujące wartości PKB:
    
    1. PKB USA (kwartalne):
       - Wartość z pierwszego miesiąca kwartału jest kopiowana na kolejne 2 miesiące
       Przykład: wartość ze stycznia -> luty, marzec
                wartość z kwietnia -> maj, czerwiec
                wartość z lipca -> sierpień, wrzesień
                wartość z października -> listopad, grudzień
       
    2. PKB globalne (roczne):
       - Wartość z danego roku jest kopiowana na wszystkie miesiące tego roku
       Przykład: wartość z 2023 -> wszystkie miesiące 2023
    """
    df = df.copy()
    
    # Konwertuj datę na datetime
    df['data'] = pd.to_datetime(df['data'])
    
    # PKB USA (kwartalne)
    if 'pkb' in df.columns:
        # Konwertuj datę na okres
        df['year'] = df['data'].dt.year
        df['month'] = df['data'].dt.month
        df['quarter'] = df['data'].dt.quarter
        
        # Sortuj po dacie
        df = df.sort_values(['year', 'month'])
        
        # Znajdź miesiące z wartościami
        valid_rows = df[pd.notna(df['pkb'])].copy()
        
        # Dla każdego miesiąca z wartością
        for _, row in valid_rows.iterrows():
            year = row['year']
            month = row['month']
            value = row['pkb']
            quarter = row['quarter']
            
            # Jeśli to pierwszy miesiąc kwartału (1, 4, 7 lub 10)
            if month % 3 == 1:
                # Znajdź dwa kolejne miesiące
                next_month1 = month + 1
                next_month2 = month + 2
                
                # Przypisz wartość do kolejnych miesięcy
                mask = (df['year'] == year) & (df['month'].isin([next_month1, next_month2]))
                df.loc[mask, 'pkb'] = value
        
        # Uzupełnij pierwsze brakujące wartości pierwszą dostępną wartością
        first_valid_pkb = df['pkb'].dropna().iloc[0]
        df['pkb'] = df['pkb'].fillna(method='ffill')
        df.loc[df['pkb'].isna(), 'pkb'] = first_valid_pkb
        
        # Uzupełnij ostatnie brakujące wartości ostatnią dostępną wartością
        df['pkb'] = df['pkb'].fillna(method='bfill')
        
        # Usuń pomocnicze kolumny
        df = df.drop(['year', 'month', 'quarter'], axis=1)
    
    # PKB globalne (roczne)
    if 'pkb_global' in df.columns:
        # Konwertuj datę na rok
        df['year'] = df['data'].dt.year
        
        # Grupuj po roku, wypełnij tą samą wartością
        df['pkb_global'] = df.groupby('year')['pkb_global'].transform(lambda x: x.iloc[0] if not x.empty else x)
        
        # Uzupełnij pierwsze brakujące wartości pierwszą dostępną wartością
        first_valid_global = df['pkb_global'].dropna().iloc[0]
        df['pkb_global'] = df['pkb_global'].fillna(method='ffill')
        df.loc[df['pkb_global'].isna(), 'pkb_global'] = first_valid_global
        
        # Uzupełnij ostatnie brakujące wartości ostatnią dostępną wartością
        df['pkb_global'] = df['pkb_global'].fillna(method='bfill')
        
        # Usuń pomocniczą kolumnę
        df = df.drop('year', axis=1)
    
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uzupełnia brakujące wartości w danych używając różnych metod:
    - Dla metali szlachetnych: interpolacja liniowa
    - Dla PKB USA i globalnego: kopiowanie wartości na odpowiednie okresy (kwartał/rok)
    - Dla pozostałych zmiennych numerycznych: SimpleImputer ze strategią mean
    """
    df = df.copy()
    
    # 1. Usuwamy kolumny, które są całkowicie puste (100% braków)
    # Kolumna jest uznawana za pustą, jeśli wszystkie wartości są NULL lub wszystkie są równe 0
    empty_cols = []
    for col in df.columns:
        if col != 'data':  # Nie sprawdzaj kolumny z datami
            if df[col].isnull().mean() == 1.0 or (df[col] == 0).mean() == 1.0:
                empty_cols.append(col)
    
    # Nie usuwaj kolumn politycznych, nawet jeśli mają same zera
    political_cols = ['niestabilność_polityczna', 'sankcje_gospodarcze']
    empty_cols = [col for col in empty_cols if col not in political_cols]
    
    df = df.drop(columns=empty_cols)
    if empty_cols:
        print(f"Usunięto całkowicie puste kolumny: {empty_cols}")

    # 2. Definiujemy kolumny z metalami
    metal_cols = ['Złoto', 'Srebro', 'Platyna', 'Pallad', 'Miedź']
    
    # 3. Interpolacja liniowa dla metali
    for col in metal_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
    
    # 4. Uzupełnianie PKB USA i globalnego
    df = fill_gdp_values(df)
    
    # 5. Dla pozostałych kolumn numerycznych używamy SimpleImputer
    other_numeric_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns 
                         if col not in metal_cols + ['pkb', 'pkb_global'] + political_cols and col in df.columns]
    
    if other_numeric_cols:
        imputer = SimpleImputer(strategy="mean")
        df[other_numeric_cols] = imputer.fit_transform(df[other_numeric_cols])
    
    return df

def standardize_data(df, columns):
    """
    Standaryzuje dane w podanych kolumnach.
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
        columns (list): Lista kolumn do standaryzacji
        
    Returns:
        pd.DataFrame: DataFrame ze standaryzowanymi danymi
    """
    from sklearn.preprocessing import StandardScaler
    
    # Skopiuj DataFrame
    df_standardized = df.copy()
    
    # Inicjalizuj StandardScaler
    scaler = StandardScaler()
    
    # Dla każdej kolumny
    for col in columns:
        if col in df.columns:  # Sprawdź czy kolumna istnieje
            # Pomiń kolumny z samymi NaN lub zerami
            if df[col].notna().any() and not (df[col] == 0).all():
                df_standardized[col] = scaler.fit_transform(df[[col]])
    
    return df_standardized

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Przetwarza dane wejściowe, skalując cechy numeryczne.
    """
    # Skopiuj DataFrame
    df_processed = df.copy()
    
    # Zapisz oryginalną kolumnę 'data', jeśli istnieje
    original_data_column = None
    if 'data' in df_processed.columns:
        original_data_column = df_processed['data'].copy()
        # Upewnij się, że jest w formacie datetime
        if not pd.api.types.is_datetime64_any_dtype(original_data_column):
             original_data_column = pd.to_datetime(original_data_column)
    
    # Lista kolumn do standaryzacji - TYLKO CECHY, NIE CELE!
    # USUNIĘTO METALE - one są celami predykcji i nie powinny być standaryzowane
    columns_to_standardize = ['inflacja', 'stopy_procentowe', 'bezrobocie', 'pkb',
                            'pkb_global', 'kurs_usd', 'indeks_vix']
    
    # Filtruj tylko kolumny, które faktycznie istnieją w dataframe
    existing_columns = [col for col in columns_to_standardize if col in df_processed.columns]
    
    scaler = StandardScaler() # Zawsze inicjalizuj scaler
    
    if existing_columns:
        # Standaryzuj dane tylko dla istniejących kolumn
        # Użyj .loc[], aby uniknąć SettingWithCopyWarning i zapewnić działanie na oryginalnym df_processed
        df_processed.loc[:, existing_columns] = scaler.fit_transform(df_processed.loc[:, existing_columns])
    
    # Przywróć oryginalną kolumnę 'data', jeśli istniała
    if original_data_column is not None and 'data' in df_processed.columns:
        df_processed['data'] = original_data_column
    elif original_data_column is not None and 'data' not in df_processed.columns:
        # Jeśli jakimś cudem kolumna 'data' została usunięta, dodaj ją z powrotem
        df_processed['data'] = original_data_column
        
    # Ostateczne sprawdzenie typu kolumny 'data'
    if 'data' in df_processed.columns and not pd.api.types.is_datetime64_any_dtype(df_processed['data']):
        # Jeśli typ nadal jest niepoprawny, spróbuj konwersji raz jeszcze
        try:
            df_processed['data'] = pd.to_datetime(df_processed['data'])
        except Exception as e:
            print(f"OSTRZEŻENIE: Nie udało się przywrócić typu datetime64 dla kolumny 'data' w preprocess_data: {e}")
            # Można tu dodać rzucenie wyjątku lub inną logikę obsługi błędu

    return df_processed, scaler

def save_processed_data(features: pd.DataFrame, output_filepath: str) -> None:
    """Zapisuje przetworzone dane do pliku CSV."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    features.to_csv(output_filepath, index=False)

def process_pipeline(input_filepath=None, output_filepath=None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Pełny pipeline przetwarzania danych:
    1. Wczytanie danych
    2. Uzupełnienie brakujących wartości
    3. Skalowanie danych
    4. Zapisanie wyników
    """
    # Ustaw domyślne ścieżki
    if input_filepath is None:
        input_filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    "data", "processed", "combined_real_and_generated_data.csv")
    if output_filepath is None:
        output_filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     "data", "processed", "processed_data.csv")

    # Wczytaj dane
    df = load_combined_data(input_filepath)
    
    # Uzupełnij brakujące wartości
    df = handle_missing_values(df)
    
    # Zapisz dane przed skalowaniem
    raw_output_filepath = os.path.join(os.path.dirname(output_filepath), "processed_data_raw.csv")
    save_processed_data(df, raw_output_filepath)
    
    # Przetwórz dane (skalowanie)
    df_processed, scaler = preprocess_data(df)
    
    # Zapisz przetworzone dane
    save_processed_data(df_processed, output_filepath)
    print(f"Zapisano przetworzone dane do: {output_filepath}")
    
    return df_processed, scaler

if __name__ == "__main__":
    # Wywołaj pipeline przetwarzania danych
    process_pipeline()
