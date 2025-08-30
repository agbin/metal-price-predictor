"""
Moduł do pobierania danych z Google Trends dotyczących zainteresowania tematami związanymi ze złotem
i innymi metalami szlachetnymi.
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from pytrends.request import TrendReq
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GoogleTrendsCollector:
    def __init__(self):
        """
        Inicjalizacja klasy do pobierania danych z Google Trends.
        """
        self.pytrends = TrendReq(hl='en-US', timeout=(10, 25), retries=2, backoff_factor=0.1)
        self.output_dir = os.path.join('data', 'external', 'google_trends')
        
        # Tworzenie katalogu wyjściowego, jeśli nie istnieje
        os.makedirs(self.output_dir, exist_ok=True)
    
    def collect_trends_data(self, keywords, start_date, end_date, geo='', cat=0, name_prefix=''):
        """
        Pobiera dane z Google Trends dla określonych słów kluczowych w danym przedziale czasowym.
        
        Args:
            keywords (list): Lista słów kluczowych do wyszukania (max 5 słów)
            start_date (str): Data początkowa w formacie YYYY-MM-DD
            end_date (str): Data końcowa w formacie YYYY-MM-DD
            geo (str): Kod kraju (np. 'US' dla USA), domyślnie pusty (globalnie)
            cat (int): Kategoria Google Trends, domyślnie 0 (wszystkie kategorie)
            name_prefix (str): Prefiks do nazwy pliku wyjściowego
            
        Returns:
            DataFrame: Dane z Google Trends
        """
        if len(keywords) > 5:
            logging.warning("Google Trends API pozwala tylko na 5 słów kluczowych naraz. Zostanie użyte pierwsze 5.")
            keywords = keywords[:5]
        
        timeframe = f'{start_date} {end_date}'
        logging.info(f"Pobieranie danych Google Trends dla słów: {keywords}, przedział: {timeframe}")
        
        try:
            # Budujemy zapytanie do API
            self.pytrends.build_payload(keywords, cat=cat, timeframe=timeframe, geo=geo)
            
            # Pobieramy dane zainteresowania w czasie
            trends_data = self.pytrends.interest_over_time()
            
            if trends_data.empty:
                logging.warning(f"Nie znaleziono danych dla podanych słów kluczowych: {keywords}")
                return None
                
            # Zapisywanie danych
            if name_prefix:
                filename = f"{name_prefix}_google_trends_{start_date}_to_{end_date}.csv"
            else:
                filename = f"google_trends_{start_date}_to_{end_date}.csv"
                
            output_path = os.path.join(self.output_dir, filename)
            trends_data.to_csv(output_path)
            logging.info(f"Dane Google Trends zapisane w: {output_path}")
            
            return trends_data
            
        except Exception as e:
            logging.error(f"Błąd podczas pobierania danych z Google Trends: {str(e)}")
            return None
    
    def get_monthly_trends(self, keywords, years_back=5, geo='', cat=0):
        """
        Pobiera miesięczne dane z Google Trends z ostatnich lat.
        Google Trends ma ograniczenia czasowe, więc dłuższe okresy są pobierane w częściach.
        
        Args:
            keywords (list): Lista słów kluczowych
            years_back (int): Ile lat wstecz pobierać dane
            geo (str): Kod kraju
            cat (int): Kategoria
            
        Returns:
            DataFrame: Połączone miesięczne dane trendów
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years_back)
        
        # Format dat dla Google Trends
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Dla dłuższych okresów, pobieramy dane po kawałku
        if years_back > 2:
            # Pobieramy dane w odcinkach po 2 lata i łączymy
            all_trends = []
            current_start = start_date
            
            while current_start < end_date:
                current_end = min(current_start + timedelta(days=730), end_date)  # 2 lata lub mniej
                
                current_start_str = current_start.strftime('%Y-%m-%d')
                current_end_str = current_end.strftime('%Y-%m-%d')
                
                trends = self.collect_trends_data(
                    keywords, 
                    current_start_str, 
                    current_end_str, 
                    geo=geo, 
                    cat=cat
                )
                
                if trends is not None and not trends.empty:
                    all_trends.append(trends)
                    
                # Dodajemy małe opóźnienie między zapytaniami
                time.sleep(1)
                
                # Przesuwamy okno czasowe
                current_start = current_end + timedelta(days=1)
                
            if all_trends:
                # Łączymy wszystkie fragmenty danych
                combined_trends = pd.concat(all_trends)
                combined_trends = combined_trends.sort_index().drop_duplicates()
                
                # Zapisujemy połączone dane
                output_path = os.path.join(
                    self.output_dir, 
                    f"combined_google_trends_{start_str}_to_{end_str}.csv"
                )
                combined_trends.to_csv(output_path)
                logging.info(f"Połączone dane Google Trends zapisane w: {output_path}")
                
                return combined_trends
            else:
                logging.warning("Nie udało się pobrać żadnych danych Google Trends")
                return None
        else:
            # Dla krótszych okresów, pobieramy wszystko naraz
            return self.collect_trends_data(
                keywords, 
                start_str, 
                end_str, 
                geo=geo, 
                cat=cat,
                name_prefix="combined"
            )

def collect_metal_trends_data(years_back=5):
    """
    Funkcja główna do zbierania danych Google Trends dotyczących metali szlachetnych.
    
    Args:
        years_back (int): Ile lat wstecz pobierać dane
        
    Returns:
        DataFrame: Dane trendów dla metali szlachetnych
    """
    collector = GoogleTrendsCollector()
    
    # Słowa kluczowe związane ze złotem
    gold_keywords = ['gold investment', 'buy gold', 'gold price', 'gold bullion', 'gold ETF']
    
    # Pobieramy dane trendów dla złota
    gold_trends = collector.get_monthly_trends(
        gold_keywords, 
        years_back=years_back, 
        geo='',  # Globalnie
        cat=0    # Wszystkie kategorie
    )
    
    # Opcjonalnie można pobrać dane dla innych metali
    # silver_keywords = ['silver investment', 'buy silver', 'silver price', 'silver bullion']
    # platinum_keywords = ['platinum investment', 'buy platinum', 'platinum price']
    
    return gold_trends

def process_trends_for_model():
    """
    Przetwarza dane z Google Trends i przygotowuje je do włączenia do modelu.
    Wylicza główny wskaźnik zainteresowania złotem, który będzie dodany jako cecha.
    
    Returns:
        DataFrame: Przetworzone dane trendów gotowe do integracji z modelem
    """
    # Ścieżka do najnowszego pliku z danymi Google Trends
    trends_dir = os.path.join('data', 'external', 'google_trends')
    if not os.path.exists(trends_dir):
        logging.error(f"Katalog z danymi Google Trends nie istnieje: {trends_dir}")
        return None
    
    # Znajdujemy najnowszy plik combined
    trend_files = [f for f in os.listdir(trends_dir) if f.startswith('combined_google_trends') and f.endswith('.csv')]
    
    if not trend_files:
        logging.error("Nie znaleziono plików z danymi Google Trends. Najpierw uruchom collect_metal_trends_data().")
        return None
    
    # Sortujemy pliki według daty modyfikacji, bierzemy najnowszy
    latest_file = sorted(trend_files, key=lambda x: os.path.getmtime(os.path.join(trends_dir, x)))[-1]
    trends_path = os.path.join(trends_dir, latest_file)
    
    try:
        # Wczytujemy dane trendów
        trends_data = pd.read_csv(trends_path, index_col=0)
        trends_data.index = pd.to_datetime(trends_data.index)
        
        # Obliczamy średnie zainteresowanie złotem (średnia z wszystkich słów kluczowych)
        trends_data['gold_interest_index'] = trends_data.mean(axis=1)
        
        # Agregujemy do danych miesięcznych, jeśli nie są jeszcze miesięczne
        monthly_trends = trends_data.resample('MS').mean()
        
        # Zapisujemy przetworzone dane
        output_path = os.path.join('data', 'processed', 'google_trends_processed.csv')
        monthly_trends.to_csv(output_path)
        logging.info(f"Przetworzone dane Google Trends zapisane w: {output_path}")
        
        return monthly_trends
        
    except Exception as e:
        logging.error(f"Błąd podczas przetwarzania danych Google Trends: {str(e)}")
        return None

if __name__ == "__main__":
    # Przykładowe uruchomienie
    logging.info("Rozpoczynanie pobierania danych Google Trends")
    collect_metal_trends_data(years_back=6)  # Pobieramy dane z 6 lat
    process_trends_for_model()  # Przetwarzamy dane do formatu dla modelu
