"""
Data Loader dla projektu predykcji cen metali.

Źródła danych:
1. Dane rzeczywiste (pobierane z API):
   - Ceny metali (Yahoo Finance)
   - Dane makroekonomiczne (FRED)
   - Kurs USD i ceny ropy (Yahoo Finance)

2. Dane przykładowe (generowane losowo):
   - Dane rynkowe
"""

import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import numpy as np
import quandl
import os
from dotenv import load_dotenv
import base64
import pandas_datareader as web
import os
import time
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    print("Uwaga: Biblioteka pytrends nie jest zainstalowana. Funkcje Google Trends nie będą dostępne.")
    print("Aby zainstalować, użyj: pip install pytrends")
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Uwaga: Biblioteka transformers nie jest zainstalowana. Funkcje analizy sentymentu nie będą dostępne.")
    print("Aby zainstalować, użyj: pip install transformers")

# Załaduj zmienne środowiskowe
load_dotenv()

def ensure_directory_structure():
    """
    Sprawdza, czy katalogi 'data', 'data/raw' i 'data/processed' istnieją, 
    i tworzy je, jeśli ich nie ma.
    """
    # Uzyskaj ścieżkę do katalogu głównego projektu (jeden poziom wyżej niż src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    directories = ["../data", "../data/raw", "../data/processed", "../data/features"]
    for directory in directories:
        dir_path = os.path.join(project_root, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Utworzono katalog: {dir_path}")
        else:
            print(f"Katalog już istnieje: {dir_path}")

# Funkcja do pobierania danych finansowych z Yahoo Finance -kurs USD
def get_financial_data(ticker, start_date="1990-01-01", end_date="2025-01-01"):
    """
    Pobiera dane finansowe z Yahoo Finance.
    
    Args:
        ticker (str): Symbol instrumentu finansowego
        start_date (str): Data początkowa w formacie YYYY-MM-DD
        end_date (str): Data końcowa w formacie YYYY-MM-DD
        
    Returns:
        pd.DataFrame: DataFrame z danymi finansowymi
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval="1mo")
    data.reset_index(inplace=True)
    data = data[["Date", "Close"]]
    data.columns = ["data", ticker]
    
    # Konwertuj datę na format YYYY-MM
    data["data"] = pd.to_datetime(data["data"]).dt.to_period("M").astype(str)
    
    return data

# Funkcja do pobierania danych makroekonomicznych z FRED i łączenia ich w jeden DataFrame
def get_macro_data(start_date="2000-01-01", end_date="2025-01-01", output_file="../data/raw/real_macro_data.csv"):
    """
    Pobiera surowe dane makroekonomiczne:
    - PKB USA (GDPC1) - kwartalne
    - Globalne PKB (NYGDPMKTPCDWLD) - roczne
    - Inflacja USA (CPIAUCSL) - miesięczne
    - Stopy procentowe USA (FEDFUNDS) - miesięczne
    - Bezrobocie USA (UNRATE) - miesięczne
    - Kurs USD (DX-Y.NYB z Yahoo Finance) - dzienne
    
    Args:
        start_date (str): Data początkowa w formacie YYYY-MM-DD
        end_date (str): Data końcowa w formacie YYYY-MM-DD
        output_file (str): Ścieżka do pliku CSV, w którym zostaną zapisane dane
        
    Returns:
        pd.DataFrame: DataFrame ze wszystkimi danymi makroekonomicznymi
    """
    # Upewnij się, że struktura katalogów istnieje
    ensure_directory_structure()
    
    # Lista wskaźników z FRED dla USA
    fred_indicators = {
        'GDPC1': ('pkb', 'Q'),         # Real Gross Domestic Product (USA) - kwartalne
        'CPIAUCSL': ('inflacja', 'M'),        # Consumer Price Index - miesięczne
        'FEDFUNDS': ('stopy_procentowe', 'M'), # Federal Funds Rate - miesięczne
        'UNRATE': ('bezrobocie', 'M')         # Unemployment Rate - miesięczne
    }
    
    # Pobierz dane z FRED dla każdego wskaźnika USA
    all_data = pd.DataFrame()
    
    # Najpierw pobierz miesięczne dane jako bazę
    monthly_indicators = {k: v for k, v in fred_indicators.items() if v[1] == 'M'}
    for indicator, (column_name, freq) in monthly_indicators.items():
        try:
            df = web.DataReader(indicator, 'fred', start_date, end_date)
            df.reset_index(inplace=True)
            df.columns = ["data", column_name]
            df["data"] = df["data"].dt.to_period("M").astype(str)
            
            if all_data.empty:
                all_data = df
            else:
                all_data = all_data.merge(df, on="data", how="outer")
        except Exception as e:
            print(f"Błąd podczas pobierania {indicator}: {e}")
    
    # Dodaj kwartalne PKB USA
    quarterly_indicators = {k: v for k, v in fred_indicators.items() if v[1] == 'Q'}
    for indicator, (column_name, freq) in quarterly_indicators.items():
        try:
            df = web.DataReader(indicator, 'fred', start_date, end_date)
            df.reset_index(inplace=True)
            df.columns = ["data", column_name]
            # Konwertuj datę do formatu YYYY-MM (pierwszy miesiąc kwartału)
            df["data"] = pd.PeriodIndex(df["data"], freq='Q').astype(str).str.replace('Q1', '-01').str.replace('Q2', '-04').str.replace('Q3', '-07').str.replace('Q4', '-10')
            
            if all_data.empty:
                all_data = df
            else:
                all_data = all_data.merge(df, on="data", how="outer")
        except Exception as e:
            print(f"Błąd podczas pobierania {indicator}: {e}")
    
    # Pobierz globalne PKB (roczne dane)
    global_gdp = get_global_gdp(start_date, end_date)
    if not global_gdp.empty:
        # Konwertuj rok na pierwszy miesiąc roku (YYYY -> YYYY-01)
        global_gdp["data"] = global_gdp["data"].apply(lambda x: f"{x}-01")
        all_data = all_data.merge(global_gdp, on="data", how="outer")
    
    # Pobierz kurs USD z Yahoo Finance (dzienne dane)
    usd_index = get_financial_data("DX-Y.NYB", start_date, end_date)  # Kurs USD (indeks dolara)
    if not usd_index.empty:
        usd_index["data"] = pd.to_datetime(usd_index["data"]).dt.to_period("M").astype(str)
        usd_index = usd_index.groupby("data")["DX-Y.NYB"].mean().reset_index()  # Średnia miesięczna
        usd_index.rename(columns={"DX-Y.NYB": "kurs_usd"}, inplace=True)
        all_data = all_data.merge(usd_index, on="data", how="outer")
    
    # Wczytaj dane o ropie z pliku CSV
    try:
        commodities_data = pd.read_csv("../data/raw/commodities.csv")
        oil_data = commodities_data[["Date", "Crude Oil petroleum"]]
        oil_data.columns = ["data", "cena_ropy"]
        oil_data["data"] = pd.to_datetime(oil_data["data"]).dt.to_period("M").astype(str)
        
        # Filtruj daty
        oil_data = oil_data[(oil_data["data"] >= start_date[:7]) & (oil_data["data"] <= end_date[:7])]
        all_data = all_data.merge(oil_data, on="data", how="outer")
    except Exception as e:
        print(f"Błąd podczas wczytywania danych o ropie: {e}")
    
    # Sortuj po dacie
    all_data = all_data.sort_values("data")
    
    # Zapisz do pliku CSV
    all_data.to_csv(output_file, index=False)
    print(f"Dane zapisano do pliku {output_file}")
    
    return all_data

def get_global_gdp(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pobiera globalne PKB z FRED.
    
    Args:
        start_date (str): Data początkowa w formacie YYYY-MM-DD
        end_date (str): Data końcowa w formacie YYYY-MM-DD
        
    Returns:
        pd.DataFrame: DataFrame z globalnym PKB (roczne wartości)
    """
    try:
        # Pobierz roczne dane
        df = web.DataReader('NYGDPMKTPCDWLD', 'fred', start_date, end_date)
        df.reset_index(inplace=True)
        df.columns = ["data", "pkb_global"]
        df["data"] = df["data"].dt.to_period("Y").astype(str)
        return df
        
    except Exception as e:
        print(f"Błąd podczas pobierania globalnego PKB: {e}")
        return pd.DataFrame()

# Funkcja do pobierania historycznych cen metali z Yahoo Finance i plików CSV
def get_metal_prices(start_date="2000-08-01", end_date="2025-01-01", output_file="../data/raw/real_metal_prices.csv"):
    """
    Pobiera historyczne ceny metali z Yahoo Finance.
    
    Metale:
    - Złoto (GLD, GC=F, XAUUSD=X)
    - Srebro (SLV, SI=F, XAGUSD=X)
    - Platyna (PL=F, PPLT)
    - Pallad (PA=F, PALL)
    - Miedź (HG=F, CPER)
    
    Args:
        start_date (str): Data początkowa w formacie YYYY-MM-DD
        end_date (str): Data końcowa w formacie YYYY-MM-DD
        output_file (str): Ścieżka do pliku CSV, w którym zostaną zapisane dane
        
    Returns:
        pd.DataFrame: DataFrame z cenami metali
    """
    # Upewnij się, że struktura katalogów istnieje
    ensure_directory_structure()
    
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
                    if len(df) > 10:  # Arbitralna wartość, możesz dostosować
                        print(f"Sukces! Pobrano {len(df)} rekordów z {symbol}")
                        break
                    else:
                        print(f"Zbyt mało danych dla {symbol} (tylko {len(df)} rekordów), próbuję następny symbol...")
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
    
    if all_data is not None:
        # Sortuj po dacie
        all_data = all_data.sort_values("data")
        
        # Pokaż statystyki
        print("\nStatystyki:")
        for col in all_data.columns:
            if col != "data":
                print(f"{col}: {len(all_data[~all_data[col].isna()])} rekordów")
        
        # Zapisz do pliku CSV
        all_data.to_csv(output_file, index=False)
        print(f"\nDane zapisano do pliku {output_file}")
    else:
        print("Nie udało się pobrać żadnych danych o metalach")
    
    return all_data

# Funkcja do pobierania danych rynkowych (indeks VIX) z Yahoo Finance
def get_market_data_vix(start_date, end_date):
    """
    Pobiera dane rynkowe (indeks VIX) z Yahoo Finance.

    Args:
        start_date (str): Data początkowa w formacie YYYY-MM-DD
        end_date (str): Data końcowa w formacie YYYY-MM-DD

    Returns:
        pd.DataFrame: DataFrame z danymi rynkowymi (indeks VIX)
    """
    try:
        # Pobierz dane VIX z Yahoo Finance
        vix_data = get_financial_data("^VIX", start_date, end_date)
        vix_data.rename(columns={"^VIX": "indeks_vix"}, inplace=True)
        return vix_data
    except Exception as e:
        print(f"Błąd podczas pobierania danych VIX: {e}")
        return pd.DataFrame()

def save_raw_data(df: pd.DataFrame, output_file: str = None) -> None:
    """
    Zapisuje surowe dane do pliku CSV.
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
        output_file (str): Ścieżka do pliku wyjściowego (opcjonalna)
    """
    if output_file is None:
        output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "data", "processed", "combined_real_and_generated_data.csv")
    
    # Upewnij się, że katalog istnieje
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Zapisz dane
    df.to_csv(output_file, index=False)
    print(f"Dane zapisano do pliku {output_file}")

def create_directories():
    """
    Tworzy wymagane katalogi, jeśli nie istnieją.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    directories = [
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "data", "raw"),
        os.path.join(base_dir, "data", "processed"),
        os.path.join(base_dir, "data", "features"),
        os.path.join(base_dir, "data", "external"),
        os.path.join(base_dir, "data", "external", "google_trends"),
        os.path.join(base_dir, "data", "external", "central_bank")
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Utworzono katalog: {directory}")
        else:
            print(f"Katalog już istnieje: {directory}")

def load_and_combine_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Ładuje i łączy wszystkie dane w jeden DataFrame.
    
    Args:
        start_date (str): Data początkowa w formacie YYYY-MM-DD
        end_date (str): Data końcowa w formacie YYYY-MM-DD
        
    Returns:
        pd.DataFrame: Połączone dane
    """
    # Pobierz dane o cenach metali
    metal_prices = get_metal_prices(start_date, end_date)
    
    # Pobierz dane makroekonomiczne
    macro_data = get_macro_data(start_date, end_date)
    
    # Pobierz dane rynkowe (VIX)
    market_data = get_market_data_vix(start_date, end_date)
    
    # Połącz wszystkie dane
    combined_data = pd.merge(metal_prices, macro_data, on='data', how='outer')
    combined_data = pd.merge(combined_data, market_data, on='data', how='outer')
    
    # Zapisz połączone dane
    save_raw_data(combined_data)
    
    return combined_data

def get_google_trends_data(keywords, start_date, end_date, geo='', cat=0):
    """
    Pobiera dane z Google Trends dla określonych słów kluczowych w danym przedziale czasowym.
    
    Args:
        keywords (list): Lista słów kluczowych do wyszukania (max 5 słów)
        start_date (str): Data początkowa w formacie YYYY-MM-DD
        end_date (str): Data końcowa w formacie YYYY-MM-DD
        geo (str): Kod kraju (np. 'US' dla USA), domyślnie pusty (globalnie)
        cat (int): Kategoria Google Trends, domyślnie 0 (wszystkie kategorie)
        
    Returns:
        pd.DataFrame: Dane z Google Trends
    """
    if not PYTRENDS_AVAILABLE:
        print("Biblioteka pytrends nie jest dostępna. Nie można pobrać danych Google Trends.")
        return None
        
    if len(keywords) > 5:
        print("Google Trends API pozwala tylko na 5 słów kluczowych naraz. Zostanie użyte pierwsze 5.")
        keywords = keywords[:5]
    
    timeframe = f'{start_date} {end_date}'
    print(f"Pobieranie danych Google Trends dla słów: {keywords}, przedział: {timeframe}")
    
    try:
        # Inicjalizujemy klienta pytrends
        pytrends = TrendReq(hl='en-US', timeout=(10, 25), retries=2, backoff_factor=0.1)
        
        # Budujemy zapytanie do API
        pytrends.build_payload(keywords, cat=cat, timeframe=timeframe, geo=geo)
        
        # Pobieramy dane zainteresowania w czasie
        trends_data = pytrends.interest_over_time()
        
        if trends_data.empty:
            print(f"Nie znaleziono danych dla podanych słów kluczowych: {keywords}")
            return None
        
        # Resetujemy indeks, aby data była jako kolumna
        trends_data.reset_index(inplace=True)
        
        # Zmieniamy nazwę kolumny z datą na 'data' dla spójności z resztą danych
        trends_data.rename(columns={'date': 'data'}, inplace=True)
        
        # Konwertujemy datę na format YYYY-MM dla spójności z resztą danych
        trends_data['data'] = pd.to_datetime(trends_data['data']).dt.to_period('M').astype(str)
        
        # Obliczamy średnie zainteresowanie złotem (średnia z wszystkich słów kluczowych)
        trends_data['gold_interest_index'] = trends_data[keywords].mean(axis=1)
        
        # Agregujemy do danych miesięcznych, jeśli nie są jeszcze miesięczne
        monthly_trends = trends_data.groupby('data').mean().reset_index()
        
        # Zapisujemy surowe dane
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'data', 'external', 'google_trends')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"google_trends_{start_date}_to_{end_date}.csv")
        monthly_trends.to_csv(output_path, index=False)
        print(f"Dane Google Trends zapisane w: {output_path}")
        
        return monthly_trends
            
    except Exception as e:
        print(f"Błąd podczas pobierania danych z Google Trends: {str(e)}")
        return None

def get_central_bank_sentiment(start_date, end_date):
    """
    Analizuje sentyment komunikatów banków centralnych.
    
    Args:
        start_date (str): Data początkowa w formacie YYYY-MM-DD
        end_date (str): Data końcowa w formacie YYYY-MM-DD
    
    Returns:
        pd.DataFrame: DataFrame z indeksem sentymentu banków centralnych
    """
    if not TRANSFORMERS_AVAILABLE:
        print("Biblioteka transformers nie jest dostępna. Nie można przeprowadzić analizy sentymentu.")
        return None
    
    # Ta funkcja jest bardziej złożona i wymaga dodatkowej implementacji
    # Poniżej jest uproszczona wersja, która tworzy dane przykładowe
    # W rzeczywistej implementacji trzeba byłoby pobrać komunikaty z witryn banków centralnych
    
    print("Uwaga: Ta funkcja obecnie generuje przykładowe dane. W rzeczywistej implementacji należy pobrać i analizować rzeczywiste komunikaty banków centralnych.")
    
    # Tworzymy przykładowy DataFrame z danymi miesięcznymi
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    df = pd.DataFrame({
        'data': [d.strftime('%Y-%m') for d in date_range],
        'cb_sentiment_index': np.random.normal(0, 1, len(date_range))  # Losowy indeks sentymentu
    })
    
    # W rzeczywistej implementacji indeks sentymentu byłby obliczany na podstawie analizy tekstu komunikatów
    
    # Zapisujemy dane
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data', 'external', 'central_bank')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"central_bank_sentiment_{start_date}_to_{end_date}.csv")
    df.to_csv(output_path, index=False)
    print(f"Dane sentymentu banków centralnych zapisane w: {output_path}")
    
    return df

def get_all_data(start_date, end_date):
    """
    Pobiera wszystkie dane i łączy je w jeden DataFrame.

    Dane rzeczywiste:
    - Ceny metali (pliki CSV)
    - Dane makroekonomiczne (FRED)
    - Kurs USD (Yahoo Finance)
    - Indeks VIX (Yahoo Finance)
    - Google Trends dla złota (jeśli dostępne)
    - Sentyment banków centralnych (jeśli dostępne)

    Args:
        start_date (str): Data początkowa w formacie YYYY-MM-DD
        end_date (str): Data końcowa w formacie YYYY-MM-DD

    Returns:
        pd.DataFrame: DataFrame ze wszystkimi danymi
    """
    try:
        # Utwórz katalogi jeśli nie istnieją
        create_directories()
        
        # Pobierz dane o cenach metali
        metal_prices = get_metal_prices(start_date, end_date)
        if metal_prices is None or metal_prices.empty:
            print("Nie udało się pobrać danych o cenach metali")
            return pd.DataFrame()
        
        # Pobierz dane makroekonomiczne
        macro_data = get_macro_data(start_date, end_date)
        
        # Pobierz dane rynkowe (VIX)
        market_data = get_market_data_vix(start_date, end_date)
        
        # Połącz wszystkie dane
        all_data = metal_prices.copy()
        
        # Dodaj dane makroekonomiczne
        if macro_data is not None and not macro_data.empty:
            all_data = all_data.merge(macro_data, on="data", how="outer")
        
        # Dodaj dane rynkowe
        if market_data is not None and not market_data.empty:
            all_data = all_data.merge(market_data, on="data", how="outer")
        
        # Pobierz dane Google Trends dla złota (jeśli biblioteka jest dostępna)
        if PYTRENDS_AVAILABLE:
            print("Pobieranie danych Google Trends dla złota...")
            gold_keywords = ['gold investment', 'buy gold', 'gold price', 'gold bullion', 'gold ETF']
            trends_data = get_google_trends_data(gold_keywords, start_date, end_date)
            
            # Dodaj dane trendów, jeśli są dostępne
            if trends_data is not None and not trends_data.empty:
                # Zostawiamy tylko kolumny 'data' i 'gold_interest_index'
                trends_data_slim = trends_data[['data', 'gold_interest_index']]
                all_data = all_data.merge(trends_data_slim, on="data", how="outer")
                print("Dane Google Trends dodane do zbioru danych.")
        else:
            print("Pomijanie pobierania danych Google Trends - biblioteka pytrends nie jest dostępna.")
        
        # Pobierz dane sentymentu banków centralnych (jeśli biblioteka jest dostępna)
        if TRANSFORMERS_AVAILABLE:
            print("Pobieranie danych sentymentu banków centralnych...")
            cb_sentiment_data = get_central_bank_sentiment(start_date, end_date)
            
            # Dodaj dane sentymentu, jeśli są dostępne
            if cb_sentiment_data is not None and not cb_sentiment_data.empty:
                all_data = all_data.merge(cb_sentiment_data, on="data", how="outer")
                print("Dane sentymentu banków centralnych dodane do zbioru danych.")
        else:
            print("Pomijanie pobierania danych sentymentu - biblioteka transformers nie jest dostępna.")
        
        # Sortuj po dacie
        all_data = all_data.sort_values("data")
        
        # Zapisz do pliku CSV
        save_raw_data(all_data)
        
        return all_data
    except Exception as e:
        print(f"Błąd podczas pobierania danych: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    ensure_directory_structure()
    
    # Sprawdź, czy biblioteki do nowych funkcjonalności są dostępne
    if not PYTRENDS_AVAILABLE:
        print("\nUWAGA: Aby używać funkcji Google Trends, zainstaluj bibliotekę pytrends:\npip install pytrends\n")
    if not TRANSFORMERS_AVAILABLE:
        print("\nUWAGA: Aby używać funkcji analizy sentymentu, zainstaluj bibliotekę transformers:\npip install transformers\n")
    
    # Pobierz wszystkie dane
    get_all_data("2000-08-01", "2025-01-01")