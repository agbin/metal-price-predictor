# System predykcji cen metali

System do prognozowania cen metali (złota, srebra, platyny, palladu, miedzi) na podstawie danych historycznych i wskaźników makroekonomicznych.

## Struktura projektu

```
metal_predictor/
│
├── data/                           # Katalog z danymi
│   ├── raw/                        # Dane surowe (nieprzetworzone)
│   │   ├── real_metal_prices.csv   # Historyczne ceny metali z Yahoo Finance
│   │   └── real_macro_data.csv     # Wskaźniki makroekonomiczne z FRED i Yahoo Finance
│   │
│   ├── processed/                  # Dane przetworzone
│   │   └── combined_real_and_generated_data.csv  # Dane połączone (metale + wskaźniki makro)
│   │
│   └── features/                   # Dane z dodanymi cechami inżynieryjnymi
│       └── engineered_features.csv # Dane z dodanymi średnimi ruchomymi, zmianami procentowymi, itp.
│
├── models/                         # Zapisane modele
│   ├── model_Złoto.pkl            # Model dla złota
│   ├── model_Srebro.pkl           # Model dla srebra
│   ├── model_Platyna.pkl          # Model dla platyny
│   └── scaler.pkl                 # Skaler do normalizacji danych
│
├── src/                            # Kod źródłowy
│   ├── data_loader.py             # Ładowanie i przygotowanie danych
│   ├── processing.py              # Przetwarzanie danych i inżynieria cech
│   ├── model_training.py          # Trenowanie modeli
│   ├── predict.py                 # Predykcja cen metali
│   ├── compare_methods.py         # Porównanie różnych metod predykcji
│   ├── exploratory_analysis.py    # Eksploracyjna analiza danych
│   └── processing_backtest.py     # Przetwarzanie danych do backtestingu
│
├── results/                        # Wyniki i wizualizacje
│
└── app.py                          # Główna aplikacja (Streamlit)
```

## Źródła danych

System wykorzystuje następujące źródła danych:

1. **Ceny metali szlachetnych i przemysłowych** (real_metal_prices.csv):
   - Złoto, srebro, platyna, pallad, miedź
   - Źródło: Yahoo Finance
   - Częstotliwość: miesięczna
   - Format: data, ceny zamknięcia

2. **Wskaźniki makroekonomiczne** (real_macro_data.csv):
   - PKB USA (kwartalne)
   - Globalne PKB (roczne)
   - Inflacja USA (miesięczna)
   - Stopy procentowe FED (miesięczne)
   - Bezrobocie USA (miesięczne)
   - Kurs USD (miesięczne)
   - Indeks VIX (miesięczne)
   - Źródła: FRED, Yahoo Finance
   - Częstotliwość: różna (interpolowane do miesięcznych)

3. **Cechy inżynieryjne** (engineered_features.csv):
   - Średnie ruchome (3, 6, 12 miesięcy)
   - Odchylenia standardowe (3, 6, 12 miesięcy)
   - Zmiany procentowe (1, 3, 6, 12 miesięcy)
   - Opóźnione wartości (lag 1, 3, 6, 12 miesięcy)

## Modele predykcyjne

System wykorzystuje następujące modele regresyjne:

- Orthogonal Matching Pursuit (OMP) - zwykle najlepszy dla metali po dodaniu autoregresji
- Ridge Regression
- ElasticNet
- Lasso Regression
- Linear Regression
- Support Vector Regression (SVR)

## Jak korzystać z systemu

### Instalacja wymaganych bibliotek

```bash
pip install -r requirements.txt
```

### Pobieranie i przetwarzanie danych

```bash
python -m src.data_loader
python -m src.processing
```

### Trenowanie modeli

```bash
python -m src.model_training
```

### Predykcja cen metali

```bash
python -m src.predict
```

### Uruchomienie aplikacji Streamlit

```bash
/home/agnieszka/miniconda3/bin/python -m streamlit run app.py
```

**Ważna uwaga:** Należy użyć pełnej ścieżki do interpretera Python z `miniconda`, aby uniknąć konfliktów z domyślnym, systemowym interpreterem Python. Użycie samego `streamlit run app.py` może prowadzić do błędów związanych z niekompatybilnymi wersjami bibliotek.

## Główne funkcjonalności

1. **Pobieranie danych** - automatyczne pobieranie najnowszych danych z API
2. **Przetwarzanie danych** - czyszczenie, interpolacja brakujących wartości, normalizacja
3. **Inżynieria cech** - tworzenie zaawansowanych cech na podstawie surowych danych
4. **Trenowanie modeli** - wybór najlepszego modelu dla każdego metalu
5. **Predykcja** - prognozowanie cen metali na przyszłe okresy
6. **Wizualizacja** - wykresy historycznych i prognozowanych cen
7. **Backtesting** - testowanie dokładności modeli na danych historycznych
8. **Interfejs użytkownika** - interaktywna aplikacja do wizualizacji prognoz

## Uwagi dotyczące dokładności prognoz

Badania wykazały, że dodanie opóźnionych wartości zmiennej docelowej (np. Price_lag_1, Price_lag_3) znacząco poprawiło dokładność predykcji (MAPE spadł z około 5% do 3-4.5%) dla różnych metali (złota, platyny, miedzi) w testach backtestingowych. Ta autoregresyjna inżynieria cech okazała się skuteczniejsza niż dodawanie opóźnień innych wskaźników makroekonomicznych.
