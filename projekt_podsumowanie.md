# Podsumowanie projektu Metal Predictor - 11.08.2025

## Główne informacje o projekcie

Metal Predictor to aplikacja do prognozowania cen metali szlachetnych (złota, srebra, platyny, palladu i miedzi) z wykorzystaniem technik uczenia maszynowego. Aplikacja pozwala na:
1. Analizę historycznych danych o cenach metali
2. Trenowanie modeli predykcyjnych
3. Generowanie prognoz cen na przyszłe okresy

## Struktura projektu

Projekt składa się z następujących głównych elementów:
- **app.py** - główna aplikacja Streamlit
- **src/** - katalog z modułami źródłowymi
- **data/raw/** - dane surowe
- **data/processed/** - przetworzone dane gotowe do modelowania
- **models/** - zapisane modele dla poszczególnych metali
- **update_metal_prices.py** - skrypt do aktualizacji danych o cenach
- **backtest/** - katalog z narzędziami do walidacji modeli

## Źródła danych

1. **Ceny metali**:
   - Yahoo Finance, tickery:
     - Złoto: `GC=F`, `XAUUSD=X`, `GLD`
     - Srebro: `SI=F`, `XAGUSD=X`, `SLV`
     - Platyna: `PL=F`, `PPLT`
     - Pallad: `PA=F`, `PALL`
     - Miedź: `HG=F`, `CPER`
   - Częstotliwość: Miesięczne uśrednienia wartości dziennych
   - Zakres czasowy: 2000-12-01 do 2025-07-01

2. **Wskaźniki makroekonomiczne (USA)**:
   - Inflacja (CPI)
   - Stopy procentowe (FED)
   - Bezrobocie
   - PKB krajowy
   - PKB globalny (Bank Światowy)
   - Kurs USD (DXY)
   - Indeks VIX

## Cechy modelu

1. **Autoregresyjne cechy**:
   - Lag_1: Wartość ceny z poprzedniego miesiąca
   - Lag_3: Wartość ceny sprzed trzech miesięcy
   - Wpływ: Redukcja MAPE z ~5% do ~3-4.5% w testach

2. **Algorytmy**:
   - Automatyczna selekcja najlepszego modelu z biblioteki PyCaret
   - Najlepsze wyniki dają:
     - Orthogonal Matching Pursuit (OMP)
     - Lasso Regression
     - Decision Tree Regressor

## Dzisiejsza praca (11.08.2025)

### Co zostało zrobione:

1. **Walidacja modelu**:
   - Stworzenie skryptu `backtest/evaluate_forecast.py` do oceny jakości prognoz
   - Trenowanie modelu na danych do 2023 i ocena prognoz na okres 2024-07.2025
   - Naprawa błędu `KeyError: 'data'` w skrypcie ewaluacyjnym
   - Wyliczenie metryk błędów (MAPE, RMSE, R²) dla każdego metalu
   - Wyniki pokazują zmienną jakość predykcji:
     - Najlepsza dla srebra (MAPE ~6.8%, R² ~0.37)
     - Średnia dla złota, platyny i miedzi (MAPE ~8-11%, ale R² negatywne)
     - Słaba dla palladu (MAPE ~29.7%, bardzo negatywne R²)

2. **Ulepszenie interfejsu użytkownika Streamlit**:
   - Stworzenie przejrzystego i minimalistycznego interfejsu
   - Dodanie szczegółowych opisów funkcjonalności i czynników modelu
   - Reorganizacja układu z użyciem zakładek i sekcji rozwijanych:
     - Główna strona z ikonami metali i minimalistycznym designem
     - System zakładek (Prognozowanie, O Aplikacji, Dokładność)
     - Rozwijane sekcje z dodatkowymi informacjami (dane, cechy, wskaźniki)
   - Doprecyzowanie pochodzenia wskaźników makroekonomicznych (głównie USA)

3. **Dokumentacja projektu**:
   - Szczegółowy opis zastosowanych metod i czynników
   - Wyjaśnienie procesu predykcji i znaczenia poszczególnych wskaźników

### Pozostałe zadania:

1. **Wdrożenie**:
   - Finalne testy aplikacji
   - Weryfikacja poprawności działania wszystkich elementów interfejsu
   - Sprawdzenie, czy wszystkie elementy UI są na właściwych miejscach
   - Integracja panelu prognozowania z nowym układem zakładek

2. **Możliwe ulepszenia modelu**:
   - Poprawienie dokładności prognoz dla palladu
   - Rozważenie dodania innych istotnych czynników rynkowych
   - Eksperymenty z innymi typami modeli uczenia maszynowego

## Kontekst wdrożenia

Aplikacja zostanie wdrożona na Digital Ocean jako kontener Docker. Przy wdrożeniu:
- Tylko interfejs Streamlit (port 8501) będzie widoczny dla użytkowników końcowych
- Kod backend, dane i modele pozostaną ukryte w kontenerze
- Przygotowano już pliki Dockerfile i .dockerignore

## Ważne uwagi

1. Schematy działania algorytmu predykcyjnego:
   - Implementacja rekurencyjnego aktualizowania cech opóźnionych (lag features)
   - Prognozowanie z krokiem do przodu z aktualizacją lag_1 i lag_3 w każdej iteracji

2. Wyniki backtestingu pokazują, że model działa najlepiej dla srebra, a najgorzej dla palladu - może to sugerować, że dla palladu potrzebne są inne cechy lub algorytmy.

3. Interfejs użytkownika został zaprojektowany z naciskiem na minimalizm i przejrzystość, z bardziej szczegółowymi informacjami dostępnymi pod zakładkami i w sekcjach rozwijanych.
