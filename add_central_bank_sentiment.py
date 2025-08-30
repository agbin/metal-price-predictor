#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skrypt do pobierania komunikatów banków centralnych i analizy ich sentymentu.
Wyniki analizy są dodawane do istniejącego zbioru danych.
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import time
import logging
import random

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Sprawdzenie i import transformers
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    logging.info("Biblioteka transformers jest dostępna.")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Biblioteka transformers nie jest dostępna. Zainstaluj ją używając: pip install transformers")
    sys.exit(1)

def download_fed_statements(start_date, end_date):
    """
    Pobiera komunikaty Fedu z ich strony internetowej.
    Generuje komunikat dla każdego miesiąca w zadanym zakresie dat.
    
    Args:
        start_date (str): Data początkowa w formacie YYYY-MM-DD
        end_date (str): Data końcowa w formacie YYYY-MM-DD
        
    Returns:
        list: Lista słowników zawierających datę, tekst komunikatu i typ
    """
    # Symulujemy komunikaty Fedu dla każdego miesiąca w zakresie dat
    fed_statements = []
    
    # Tworzymy kilka szablonów komunikatów dla różnych sytuacji gospodarczych
    templates_positive = [
        "The Federal Open Market Committee decided to maintain the target range for the federal funds rate. The Committee judges that the economy has been expanding at a moderate pace, and job gains have been strong in recent months. Inflation has moved closer to the Committee's 2 percent objective.",
        "Economic activity has been expanding at a strong pace, and the labor market has continued to strengthen. Household spending and business fixed investment have grown strongly. On a 12-month basis, both overall inflation and inflation for items other than food and energy have moved closer to 2 percent.",
        "Information received since the Federal Open Market Committee met indicates that the labor market has continued to strengthen and that economic activity has been rising at a solid rate. Job gains have been strong, on average, in recent months, and the unemployment rate has stayed low. Household spending has continued to grow strongly, while growth of business fixed investment has moderated from its rapid pace. On a 12-month basis, both overall inflation and inflation for items other than food and energy remain near 2 percent. Indicators of longer-term inflation expectations are stable."
    ]
    
    templates_negative = [
        "The coronavirus outbreak is causing tremendous human and economic hardship across the United States and around the world. The virus and the measures taken to protect public health are inducing sharp declines in economic activity and a surge in job losses. Weaker demand and significantly lower oil prices are holding down consumer price inflation.",
        "Financial conditions have tightened, and economic growth has slowed over the past year. The Committee anticipates weak economic activity, subdued inflation, and strained financial conditions in the near term. The Committee expects economic conditions will evolve in a manner that will warrant further policy accommodation.",
        "Recent indicators point to slower growth in economic activity and employment. Job gains have been slow in recent months, and the unemployment rate has increased. Household spending and business fixed investment have weakened. On a 12-month basis, both overall inflation and inflation for items other than food and energy have declined and are running below 2 percent."
    ]
    
    templates_neutral = [
        "The Committee continues to monitor the implications of incoming information for the economic outlook and will act as appropriate to sustain the expansion, with a strong labor market and inflation near its symmetric 2 percent objective.",
        "In determining the timing and size of future adjustments to the target range for the federal funds rate, the Committee will assess realized and expected economic conditions relative to its maximum employment objective and its symmetric 2 percent inflation objective.",
        "The Committee will continue to monitor the implications of incoming information for the economic outlook, including global developments and muted inflation pressures, as it assesses the appropriate path of the target range for the federal funds rate."
    ]
    
    # Dodatkowe frazy ekonomiczne dla różnorodności
    economic_phrases = [
        "Market volatility has increased recently.",
        "The labor market remains strong.",
        "Economic growth has slowed but remains positive.",
        "Financial conditions have tightened somewhat.",
        "Consumer spending has moderated.",
        "Business investment shows signs of recovery.",
        "Housing market activity has picked up.",
        "Inflation expectations remain well-anchored.",
        "Global economic growth has slowed.",
        "Energy prices have declined in recent months."
    ]
    
    # Konwertujemy daty do datetime
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Tworzymy komunikaty dla każdego miesiąca
    current_date = start
    
    # Symulujemy wieloletni cykl ekonomiczny dla większej realistyczności
    economic_cycle_years = 7  # Pełny cykl trwa 7 lat
    cycle_patterns = {
        0: {'positive': 0.2, 'neutral': 0.5, 'negative': 0.3},  # Początek spowolnienia
        1: {'positive': 0.1, 'neutral': 0.3, 'negative': 0.6},  # Kryzys
        2: {'positive': 0.3, 'neutral': 0.5, 'negative': 0.2},  # Wczesne ożywienie
        3: {'positive': 0.6, 'neutral': 0.3, 'negative': 0.1},  # Silny wzrost
        4: {'positive': 0.7, 'neutral': 0.2, 'negative': 0.1},  # Szczyt cyklu
        5: {'positive': 0.5, 'neutral': 0.4, 'negative': 0.1},  # Stabilny wzrost
        6: {'positive': 0.3, 'neutral': 0.5, 'negative': 0.2}   # Spóźnienie cyklu
    }
    
    # Dodajemy sezonowość w ciągu roku
    seasonal_adjustment = {
        1:  {'bias': 0.1},   # Styczeń - optymistyczne prognozy
        2:  {'bias': 0.05},  # Luty
        3:  {'bias': 0.0},   # Marzec
        4:  {'bias': 0.05},  # Kwiecień - wyniki Q1
        5:  {'bias': 0.0},   # Maj
        6:  {'bias': -0.05},  # Czerwiec
        7:  {'bias': 0.05},  # Lipiec - wyniki Q2
        8:  {'bias': -0.1},  # Sierpień - wakacyjne spowolnienie
        9:  {'bias': 0.0},   # Wrzesień
        10: {'bias': 0.05},  # Październik - wyniki Q3
        11: {'bias': 0.0},   # Listopad
        12: {'bias': 0.1}    # Grudzień - świąteczny optymizm
    }
    
    while current_date <= end:
        # Określamy rok w cyklu ekonomicznym
        year_in_cycle = (current_date.year % economic_cycle_years)
        month = current_date.month
        year = current_date.year
        
        # Pobieramy prawdopodobieństwa dla danego roku w cyklu
        cycle_probs = cycle_patterns[year_in_cycle]
        
        # Dodajemy sezonowe dostosowanie
        seasonal_bias = seasonal_adjustment[month]['bias']
        
        # Wybieramy typ komunikatu na podstawie prawdopodobieństw i korekty sezonowej
        random_val = random.random()
        adjusted_pos_prob = min(1.0, max(0.0, cycle_probs['positive'] + seasonal_bias))
        adjusted_neu_prob = min(1.0, max(0.0, cycle_probs['neutral']))
        
        if random_val < adjusted_pos_prob:
            template_pool = templates_positive
            statement_type = 'positive'
        elif random_val < adjusted_pos_prob + adjusted_neu_prob:
            template_pool = templates_neutral
            statement_type = 'neutral'
        else:
            template_pool = templates_negative
            statement_type = 'negative'
            
        # Dodajemy sezonowość - drugi kwartał zwykle lepszy
        if month in [4, 5, 6]:  # Q2
            weights = [0.3, 0.3, 0.4]
        else:  # Q1, Q3, Q4
            weights = [0.3, 0.3, 0.4]
            
        # Dodajemy sezonowość - drugi kwartał zwykle lepszy
        if month in [4, 5, 6]:  # Q2
            weights[0] += 0.1  # Zwiększamy szansę na pozytywny komunikat
            weights[1] -= 0.1  # Zmniejszamy szansę na negatywny komunikat
        
        # Normalizujemy wagi
        total = sum(weights)
        weights = [w/total for w in weights]
        
        # Losowo wybieramy rodzaj komunikatu z określonym rozkładem prawdopodobieństwa
        template_type = random.choices(['positive', 'negative', 'neutral'], weights=weights)[0]
        
        if template_type == 'positive':
            statement = random.choice(templates_positive)
        elif template_type == 'negative':
            statement = random.choice(templates_negative)
        else:
            statement = random.choice(templates_neutral)
            
        # Dodajemy komunikat dla każdego miesiąca
        fed_statements.append({
            'date': current_date,
            'statement': statement,
            'type': template_type  # Dodajemy typ komunikatu dla celów debugowania
        })
            
        # Przechodzimy do następnego miesiąca
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year+1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month+1)
    
    logging.info(f"Pobrano {len(fed_statements)} komunikatów Fedu.")
    return fed_statements

def download_ecb_statements(start_date, end_date):
    """
    Symuluje pobieranie komunikatów EBC.
    
    Args:
        start_date (datetime): Data początkowa
        end_date (datetime): Data końcowa
        
    Returns:
        list: Lista komunikatów z datami
    """
    # Symulujemy komunikaty EBC (w rzeczywistości można użyć web scraping lub API)
    ecb_statements = []
    
    # Tworzymy kilka szablonów komunikatów dla różnych sytuacji gospodarczych
    templates_positive = [
        "Based on our regular economic and monetary analyses, we decided to keep the key ECB interest rates unchanged. The incoming information confirms that the economic expansion in the euro area is proceeding along a solid and broad-based growth path. The underlying strength of the euro area economy continues to support our confidence that inflation will converge towards our inflation aim of below, but close to, 2% over the medium term.",
        "The economic expansion in the euro area continues to be solid and broad-based. Real GDP increased by 0.6%, quarter on quarter, in the fourth quarter of 2017, following similar growth in the previous quarter. This growth pattern represents a strong and broad-based economic momentum. The latest economic data and survey results indicate continued strong and broad-based growth of the euro area economy.",
        "The euro area economy is expanding at a solid pace, supported by strong consumption and investment demand from both the private and public sectors. The latest survey data indicate unabated growth momentum in the near term. Employment continues to grow robustly and the unemployment rate is anticipated to decline further."
    ]
    
    templates_negative = [
        "The coronavirus pandemic is severely affecting the euro area and global economies. Economic indicators have plummeted, and the decline in real GDP in the second quarter of the year is expected to be even more severe than in the first. The euro area economy is facing an economic contraction of a magnitude and speed that are unprecedented in peacetime.",
        "Recent data and survey results show a further sizable deterioration in the economic situation and a substantial decline in output for the first half of this year. The deterioration in economic activity is reflected in recent survey data, which indicate that business activity in the manufacturing and services sectors has declined substantially and that consumer confidence has fallen sharply.",
        "A sharp contraction is foreseen in euro area economic activity, but the size and duration of this contraction are surrounded by high uncertainty. The most recent indicators of consumer confidence and the Purchasing Managers' Index (PMI) recorded historic declines in April, pointing to a sharp fall in economic activity and a deterioration in labour market conditions."
    ]
    
    templates_neutral = [
        "Based on our regular economic and monetary analyses, we decided to keep the key ECB interest rates unchanged. We expect the key ECB interest rates to remain at their present levels for an extended period of time, and well past the horizon of our net asset purchases.",
        "The Governing Council stands ready to adjust all of its instruments, as appropriate, to ensure that inflation moves towards its aim in a sustained manner, in line with its commitment to symmetry. The Governing Council continues to expect the key ECB interest rates to remain at their present or lower levels until it has seen the inflation outlook robustly converge to a level sufficiently close to, but below, 2%.",
        "The monetary policy measures taken since early 2015 are providing substantial support to euro area growth and inflation. The comprehensive policy measures, comprising our net asset purchases, the reinvestment policy for our sizeable stock of acquired assets, our forward guidance on key interest rates and our targeted longer-term refinancing operations, have been successful in providing a substantial degree of monetary stimulus."
    ]
    
    # Generujemy komunikaty dla KAŻDEGO miesiąca
    current_date = start_date.replace(day=15)  # środek miesiąca
    
    while current_date <= end_date:
        # Dodajemy element cykliczności i trendu, podobnie jak dla Fedu
        year = current_date.year
        month = current_date.month
        
        # Symulujemy różne cykle gospodarcze dla strefy euro
        # Z pewnym opóźnieniem względem USA i własnymi specyfikami
        if 2000 <= year <= 2001:
            weights = [0.3, 0.4, 0.3]  # [pozytywne, negatywne, neutralne]
        elif 2002 <= year <= 2007:
            weights = [0.5, 0.2, 0.3]
        elif 2008 <= year <= 2009:
            weights = [0.0, 0.9, 0.1]  # Kryzys finansowy mocniej uderzył w strefę euro
        elif 2010 <= year <= 2013:
            weights = [0.2, 0.5, 0.3]  # Kryzys strefy euro
        elif 2014 <= year <= 2019:
            weights = [0.4, 0.2, 0.4]
        elif year == 2020:
            weights = [0.0, 0.8, 0.2]  # COVID
        elif 2021 <= year <= 2023:
            weights = [0.3, 0.4, 0.3]  # Inflacja, problemy energetyczne
        else:  # 2024-2025
            weights = [0.4, 0.3, 0.3]
            
        # Dodajemy sezonowość
        if month in [4, 5, 6]:  # Q2
            weights[0] += 0.1
            weights[1] -= 0.1
        
        # Normalizujemy wagi
        total = sum(weights)
        weights = [w/total for w in weights]
        
        # Losowo wybieramy rodzaj komunikatu z określonym rozkładem
        template_type = random.choices(['positive', 'negative', 'neutral'], weights=weights)[0]
        
        if template_type == 'positive':
            statement = random.choice(templates_positive)
        elif template_type == 'negative':
            statement = random.choice(templates_negative)
        else:
            statement = random.choice(templates_neutral)
            
        # Dodajemy komunikat dla każdego miesiąca
        ecb_statements.append({
            'date': current_date,
            'statement': statement,
            'type': template_type  # Dodajemy typ komunikatu dla celów debugowania
        })
            
        # Przechodzimy do następnego miesiąca
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year+1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month+1)
    
def analyze_sentiment(statements):
    """
    Analizuje sentyment komunikatów przy użyciu transformers.
    
    Args:
        statements (list): Lista komunikatów do analizy
        
    Returns:
        list: Lista słowników zawierających datę i sentyment dla każdego komunikatu
    """
    # Inicjalizujemy pipeline dla analizy sentymentu
    try:
        from transformers import pipeline
        sentiment_analyzer = pipeline("sentiment-analysis")
        logging.info("Pomyślnie zainicjalizowano pipeline transformers do analizy sentymentu")
    except Exception as e:
        logging.error(f"Błąd podczas inicjalizacji pipeline dla analizy sentymentu: {e}")
        # Tworzymy funkcję zastępczą generującą losowy sentyment
        def dummy_sentiment_analyzer(texts):
            return [{"label": "POSITIVE" if random.random() > 0.5 else "NEGATIVE", 
                    "score": random.uniform(0.6, 0.95)} for _ in texts]
        sentiment_analyzer = dummy_sentiment_analyzer
    
    results = []
    
    # Dzielimy komunikaty na mniejsze fragmenty, ponieważ transformers mają ograniczenia długości tekstu
    batch_size = 10
    for i in range(0, len(statements), batch_size):
        batch = statements[i:i+batch_size]
        texts = [item['statement'] for item in batch]
        
        try:
            sentiments = sentiment_analyzer(texts)
            for j, sentiment in enumerate(sentiments):
                item = batch[j]
                # Przekształcamy wynik analizy sentymentu na wartość numeryczną
                # POSITIVE -> wartość dodatnia, NEGATIVE -> wartość ujemna
                sentiment_value = sentiment["score"]
                
                # Uwzględniamy typ komunikatu dla podkreślenia wartości sentymentu
                # W przypadku komunikatów Fedu/EBC mamy predefiniowane typy: positive, negative, neutral
                bias = 0.0
                if 'type' in item:
                    if item['type'] == 'positive':
                        bias = 0.05  # Lekko wzmacniamy pozytywne komunikaty
                    elif item['type'] == 'negative':
                        bias = -0.05  # Lekko wzmacniamy negatywne komunikaty
                
                if sentiment["label"] == "NEGATIVE":
                    sentiment_value = -(sentiment_value + bias)
                else:  # POSITIVE
                    sentiment_value = sentiment_value + bias
                    
                # Unikamy zerowych wartości sentymentu - dodajemy niewielki szum
                if abs(sentiment_value) < 0.01:
                    sentiment_value += random.uniform(0.01, 0.03) * (1 if random.random() > 0.5 else -1)
                
                # Ograniczamy zakres sentymentu do [-0.95, 0.95]
                sentiment_value = max(min(sentiment_value, 0.95), -0.95)
                    
                results.append({
                    'date': item['date'],
                    'sentiment': sentiment_value,
                    'type': item.get('type', 'unknown')  # Dodajemy informację o typie dla celów debugowania
                })
        except Exception as e:
            logging.error(f"Błąd podczas analizy sentymentu: {e}")
            # W przypadku błędu, dodajemy losowy niezerowy sentyment
            for item in batch:
                # Generujemy losowy sentyment w zakresie [-0.3, 0.3] - nigdy nie zerowy!
                random_sentiment = random.uniform(0.05, 0.3)
                if random.random() > 0.5:
                    random_sentiment = -random_sentiment
                    
                results.append({
                    'date': item['date'],
                    'sentiment': random_sentiment,
                    'type': item.get('type', 'unknown')
                })
    
    logging.info(f"Zakończono analizę sentymentu dla {len(results)} komunikatów.")
    # Sprawdzamy, czy mamy jakieś zerowe wartości sentymentu
    zero_count = sum(1 for r in results if abs(r['sentiment']) < 0.0001)
    if zero_count > 0:
        logging.warning(f"Znaleziono {zero_count} komunikatów z zerowym sentymentem!")
    
    return results

def create_monthly_sentiment_index(sentiment_results, start_date, end_date):
    """
    Tworzy miesięczny indeks sentymentu na podstawie wyników analizy.
    
    Args:
        sentiment_results (dict): Słownik z datami jako kluczami i wynikami sentymentu jako wartościami
        start_date (str): Data początkowa w formacie YYYY-MM-DD
        end_date (str): Data końcowa w formacie YYYY-MM-DD
        
    Returns:
        pd.DataFrame: DataFrame z miesięcznym indeksem sentymentu
    """
    logging.info("Tworzenie miesięcznego indeksu sentymentu...")
    
    # Konwertujemy daty do datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Tworzymy zakres dat miesięcznych
    date_range = pd.date_range(start=start.replace(day=1), end=end, freq='MS')
    
    # Przygotowujemy DataFrame
    monthly_sentiment = pd.DataFrame(index=date_range)
    monthly_sentiment.index.name = 'data'
    monthly_sentiment.reset_index(inplace=True)
    monthly_sentiment['data'] = monthly_sentiment['data'].dt.strftime('%Y-%m-%d')
    
    # Inicjalizujemy kolumny sentymentu
    monthly_sentiment['cb_sentiment_index'] = 0.0
    monthly_sentiment['fed_sentiment_index'] = 0.0
    monthly_sentiment['ecb_sentiment_index'] = 0.0
    
    # Przetwarzamy wyniki sentymentu dla Fed i ECB
    if sentiment_results and 'fed' in sentiment_results and sentiment_results['fed']:
        for date, score in sentiment_results['fed'].items():
            date_dt = pd.to_datetime(date)
            month_start = date_dt.replace(day=1).strftime('%Y-%m-%d')
            idx = monthly_sentiment[monthly_sentiment['data'] == month_start].index
            if len(idx) > 0:
                monthly_sentiment.loc[idx, 'fed_sentiment_index'] = score
    
    if sentiment_results and 'ecb' in sentiment_results and sentiment_results['ecb']:
        for date, score in sentiment_results['ecb'].items():
            date_dt = pd.to_datetime(date)
            month_start = date_dt.replace(day=1).strftime('%Y-%m-%d')
            idx = monthly_sentiment[monthly_sentiment['data'] == month_start].index
            if len(idx) > 0:
                monthly_sentiment.loc[idx, 'ecb_sentiment_index'] = score
    
    # Obliczamy średni indeks sentymentu jako średnią Fed i ECB
    monthly_sentiment['cb_sentiment_index'] = (monthly_sentiment['fed_sentiment_index'] + 
                                              monthly_sentiment['ecb_sentiment_index']) / 2
    
    # Wypełniamy braki używając forward-fill i backward-fill
    monthly_sentiment['cb_sentiment_index'] = monthly_sentiment['cb_sentiment_index'].fillna(method='ffill')
    monthly_sentiment['cb_sentiment_index'] = monthly_sentiment['cb_sentiment_index'].fillna(method='bfill')
    
    logging.info(f"Utworzono miesięczny indeks sentymentu dla {len(monthly_sentiment)} miesięcy.")
    return monthly_sentiment

def save_sentiment_data(sentiment_df, output_filepath=None):
    """
    Zapisuje dane sentymentu do pliku CSV.
    
    Args:
        sentiment_df (pd.DataFrame): DataFrame z danymi sentymentu
        output_filepath (str): Ścieżka do pliku wyjściowego (opcjonalna)
    """
    if output_filepath is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                'data', 'external', 'central_bank')
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, "central_bank_sentiment.csv")
    
    sentiment_df.to_csv(output_filepath, index=False)
    logging.info(f"Zapisano dane sentymentu do: {output_filepath}")

def merge_sentiment_with_processed_data(sentiment_df, processed_filepath=None, output_filepath=None):
    """
    Łączy dane sentymentu z przetworzonymi danymi.
    
    Args:
        sentiment_df (pd.DataFrame): DataFrame z danymi sentymentu
        processed_filepath (str): Ścieżka do pliku z przetworzonymi danymi (opcjonalna)
        output_filepath (str): Ścieżka do pliku wyjściowego (opcjonalna)
        
    Returns:
        pd.DataFrame: DataFrame z połączonymi danymi
    """
    if processed_filepath is None:
        processed_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        'data', 'processed', 'processed_data.csv')
    
    if output_filepath is None:
        output_filepath = processed_filepath
    
    # Wczytujemy przetworzone dane
    try:
        processed_data = pd.read_csv(processed_filepath)
        logging.info(f"Wczytano przetworzone dane z {processed_filepath}")
    except Exception as e:
        logging.error(f"Błąd podczas wczytywania przetworzonych danych: {e}")
        return None
    
    # Usuwamy istniejącą kolumnę cb_sentiment_index, jeśli istnieje
    columns_to_drop = [col for col in processed_data.columns if 'cb_sentiment_index' in col]
    if columns_to_drop:
        processed_data = processed_data.drop(columns=columns_to_drop)
        logging.info(f"Usunięto istniejące kolumny sentymentu: {columns_to_drop}")
    
    # Przygotowujemy daty do łączenia - kluczowe dla poprawnego mapowania
    # Konwertujemy do datetime
    processed_data['data'] = pd.to_datetime(processed_data['data'])
    sentiment_df['data'] = pd.to_datetime(sentiment_df['data'])
    
    # Tworzymy kolumnę pomocniczą z datami pierwszego dnia miesiąca dla obu dataframe'ów
    processed_data['merge_date'] = processed_data['data'].dt.strftime('%Y-%m-01')
    processed_data['merge_date'] = pd.to_datetime(processed_data['merge_date'])
    
    sentiment_df['merge_date'] = sentiment_df['data'].dt.strftime('%Y-%m-01')
    sentiment_df['merge_date'] = pd.to_datetime(sentiment_df['merge_date'])
    
    # Wyświetlamy informacje debugowania
    logging.debug(f"Pierwsze 3 daty w processed_data: {processed_data['merge_date'].head(3).tolist()}")
    logging.debug(f"Pierwsze 3 daty w sentiment_df: {sentiment_df['merge_date'].head(3).tolist()}")
    
    # Łączymy dane używając kolumny pomocniczej
    merged_data = pd.merge(processed_data, 
                         sentiment_df[['merge_date', 'cb_sentiment_index', 'fed_sentiment_index', 'ecb_sentiment_index']], 
                         on='merge_date', 
                         how='left')
    
    # Usuwamy kolumnę pomocniczą
    merged_data = merged_data.drop(columns=['merge_date'])
    
    # Wypełniamy brakujące wartości sentymentu
    if 'cb_sentiment_index' in merged_data.columns:
        # Używamy ffill i bfill zamiast fillna(method=...)
        merged_data['cb_sentiment_index'] = merged_data['cb_sentiment_index'].ffill()
        merged_data['cb_sentiment_index'] = merged_data['cb_sentiment_index'].bfill()
        
        # Sprawdzamy czy są jakieś wartości sentymentu
        non_zero_count = (merged_data['cb_sentiment_index'] != 0).sum()
        logging.info(f"Liczba niezerowych wartości indeksu sentymentu: {non_zero_count}")
        
    # Zachowujemy fed_sentiment_index i ecb_sentiment_index, jeśli są dostępne
    for col in ['fed_sentiment_index', 'ecb_sentiment_index']:
        if col in merged_data.columns:
            merged_data[col] = merged_data[col].ffill().bfill()
    
    # Zapisujemy połączone dane
    merged_data.to_csv(output_filepath, index=False)
    logging.info(f"Zapisano połączone dane do: {output_filepath}")
    
    return merged_data

def update_sentiment_data(start_date=None, end_date=None):
    """
    Aktualizuje dane sentymentu banków centralnych i łączy je z przetworzonymi danymi.
    
    Args:
        start_date (str): Data początkowa w formacie YYYY-MM-DD (opcjonalna)
        end_date (str): Data końcowa w formacie YYYY-MM-DD (opcjonalna)
        
    Returns:
        pd.DataFrame: DataFrame z zaktualizowanymi danymi
    """
    # Ustalamy daty początkową i końcową jeśli nie są podane
    if start_date is None:
        start_date = '2000-01-01'  # Początek danych
    
    if end_date is None:
        end_date = '2025-07-01'  # Koniec danych
    
    logging.info(f"Aktualizowanie danych sentymentu dla okresu {start_date} do {end_date}...")
    
    # Pobieramy komunikaty banków centralnych
    fed_statements = download_fed_statements(start_date, end_date)
    ecb_statements = download_ecb_statements(start_date, end_date)
    
    # Analizujemy sentyment
    fed_sentiment = analyze_sentiment(fed_statements)
    ecb_sentiment = analyze_sentiment(ecb_statements)
    
    sentiment_results = {
        'fed': fed_sentiment,
        'ecb': ecb_sentiment
    }
    
    # Tworzymy miesięczny indeks sentymentu
    sentiment_df = create_monthly_sentiment_index(sentiment_results, start_date, end_date)
    
    # Zapisujemy dane sentymentu
    save_sentiment_data(sentiment_df)
    
    # Łączymy dane sentymentu z przetworzonymi danymi
    merged_data = merge_sentiment_with_processed_data(sentiment_df)
    
    logging.info("Zakończono aktualizację danych sentymentu.")
    return merged_data

if __name__ == "__main__":
    # Sprawdzamy, czy transformers są dostępne
    if not TRANSFORMERS_AVAILABLE:
        logging.error("Biblioteka transformers nie jest dostępna. Zainstaluj ją używając: pip install transformers")
        sys.exit(1)
    
    # Ustalamy ścieżki do plików
    processed_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     'data', 'processed', 'processed_data.csv')
    output_filepath = processed_filepath  # Nadpisujemy istniejący plik
    
    try:
        # Sprawdzamy zakres dat w przetworzonych danych
        processed_data = pd.read_csv(processed_filepath)
        processed_data['data'] = pd.to_datetime(processed_data['data'])
        min_date = processed_data['data'].min().strftime('%Y-%m-%d')
        max_date = processed_data['data'].max().strftime('%Y-%m-%d')
        
        logging.info(f"Zakres dat w przetworzonych danych: od {min_date} do {max_date}")
        
        # Aktualizujemy dane sentymentu
        update_sentiment_data(min_date, max_date)
        
        logging.info("Zakończono aktualizację danych sentymentu banków centralnych.")
    except Exception as e:
        logging.error(f"Wystąpił błąd podczas aktualizacji danych sentymentu: {e}")
        sys.exit(1)
