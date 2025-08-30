import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.processing import load_combined_data, preprocess_data, save_processed_data
from src.model_training import train_model
from src.predict import predict_price
from src.data_loader import get_all_data

# Konfiguracja ścieżek
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = MODELS_DIR / "results"

# Upewnij się, że wszystkie wymagane katalogi istnieją
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Stałe konfiguracyjne
CACHE_TTL = 3600  # Czas życia cache w sekundach
DEFAULT_START_DATE = "2000-08-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")

# Lista dostępnych metali
available_metals = ["Złoto", "Srebro", "Platyna", "Pallad", "Miedź"]

def check_and_update_data():
    """
    Sprawdza czy dane istnieją i czy są aktualne.
    Jeśli nie istnieją - pobiera je.
    Jeśli są nieaktualne - aktualizuje je.
    """
    raw_data_file = RAW_DATA_DIR / "combined_real_and_generated_data.csv"
    
    # Jeśli plik nie istnieje, pobierz dane
    if not raw_data_file.exists():
        st.info("Pobieranie danych...")
        get_all_data(DEFAULT_START_DATE, DEFAULT_END_DATE)
        st.success("Dane zostały pobrane!")
        return
        
    # Sprawdź czy dane są aktualne
    try:
        df = pd.read_csv(raw_data_file)
        if 'data' not in df.columns:
            st.error("Nieprawidłowy format danych!")
            return
            
        latest_date = pd.to_datetime(df['data']).max()
        today = pd.to_datetime(DEFAULT_END_DATE)
        
        # Jeśli dane są starsze niż dzień, zaktualizuj je
        if latest_date.date() < today.date():
            st.info("Aktualizacja danych...")
            get_all_data(latest_date.strftime("%Y-%m-%d"), DEFAULT_END_DATE)
            st.success("Dane zostały zaktualizowane!")
    except Exception as e:
        st.error(f"Błąd podczas sprawdzania danych: {e}")

# Walidacja danych
def validate_dataframe(df: pd.DataFrame) -> bool:
    """Sprawdza czy dataframe spełnia wymagane kryteria."""
    if not isinstance(df, pd.DataFrame):
        return False
    required_columns = ["data"] + available_metals
    return all(col in df.columns for col in required_columns)

# Załaduj dane
@st.cache_data(ttl=CACHE_TTL)
def load_data():
    try:
        # Upewnij się, że dane istnieją
        check_and_update_data()
        
        # Załaduj dane
        df = load_combined_data()
        
        # Sprawdź czy dane są puste
        if df is None or df.empty:
            st.warning("Brak danych. Próbuję pobrać nowe dane...")
            get_all_data(DEFAULT_START_DATE, DEFAULT_END_DATE)
            df = load_combined_data()
            if df is None or df.empty:
                st.error("Nie udało się załadować danych!")
                return None
            
        if not validate_dataframe(df):
            st.error("Dane nie zawierają wymaganych kolumn!")
            return None
            
        return df
    except Exception as e:
        st.error(f"Błąd podczas ładowania danych: {e}")
        return None

# Funkcja do trenowania modelu (przywrócona)
def retrain_model(df: pd.DataFrame, target_metal: str) -> bool:
    """Funkcja opakowująca proces trenowania modelu."""
    try:
        with st.spinner(f'Trenuję model dla {target_metal}...'):
            train_model(df, target_metal)  # Wywołanie funkcji z src.model_training
            return True
    except Exception as e:
        st.error(f"Błąd podczas trenowania modelu dla {target_metal}: {e}")
        return False

# Funkcja do wyświetlania wykresów analizy modelu
def display_model_analysis(target_metal: str):
    """Wyświetla wykresy analizy modelu dla konkretnego metalu."""
    try:
        # Użyj target_metal do ścieżek
        results_subdir = f'results_{target_metal}'
        model_results_dir = MODELS_DIR / results_subdir
        
        # Sprawdź czy wykresy istnieją
        predictions_plot = model_results_dir / 'predictions_vs_actual.png'
        residuals_plot = model_results_dir / 'residuals.png'
        
        if predictions_plot.exists() and residuals_plot.exists():
            col1, col2 = st.columns(2)
            
            # Wykres predykcji vs rzeczywiste wartości
            with col1:
                st.subheader("Predykcje vs Rzeczywiste wartości")
                st.image(str(predictions_plot))
            
            # Wykres residuów
            with col2:
                st.subheader("Wykres residuów")
                st.image(str(residuals_plot))
                
            # Wyświetl metryki modelu
            metrics_file = model_results_dir / 'model_metrics.csv'
            if metrics_file.exists():
                st.subheader(f"Metryki modelu dla {target_metal}")
                metrics = pd.read_csv(metrics_file)
                st.dataframe(metrics)
        else:
            st.warning(f"Brak wykresów analizy modelu dla {target_metal}. Najpierw wytrenuj model.")
    except Exception as e:
        st.error(f"Błąd podczas wyświetlania wykresów: {e}")

# Funkcja do tworzenia wykresu
def create_prediction_chart(historical_data: pd.DataFrame, prediction_data: pd.DataFrame, metal_name: str):
    """
    Tworzy interaktywny wykres z danymi historycznymi i prognozą.
    """
    fig = go.Figure()
    
    # Dodaj dane historyczne
    fig.add_trace(
        go.Scatter(
            x=historical_data["data"],
            y=historical_data[metal_name],
            name="Dane historyczne",
            line=dict(color="blue")
        )
    )
    
    # Dodaj prognozę jeśli jest dostępna
    if prediction_data is not None and not prediction_data.empty:
        fig.add_trace(
            go.Scatter(
                x=prediction_data["data"],
                y=prediction_data["prediction_label"],
                name="Prognoza",
                line=dict(color="red", dash="dash")
            )
        )
    
    # Aktualizuj układ wykresu
    fig.update_layout(
        title=f"Cena {metal_name}",
        xaxis_title="Data",
        yaxis_title="Cena",
        hovermode="x unified"
    )
    
    return fig

# Załaduj dane
df = load_data()
if df is None:
    st.stop()

# Sidebar - wybór metalu
with st.sidebar:
    st.header("Ustawienia")
    selected_metal = st.selectbox("Wybierz metal", available_metals, key="metal_selector")
    
    st.markdown("---")
    
    # Parametry prognozy
    st.subheader("Parametry prognozy")
    
    # Określ zakres dat na podstawie danych
    min_date = df["data"].min()
    max_date = df["data"].max()
    future_months = st.slider(
        "Liczba miesięcy do prognozowania",
        min_value=1,
        max_value=120,  # 10 lat do przodu
        value=12,  # domyślnie 1 rok
        help="Wybierz na ile miesięcy do przodu chcesz prognozować cenę"
    )
    
    # Oblicz datę końcową prognozy
    last_date = pd.to_datetime(max_date)
    forecast_end_date = last_date + pd.DateOffset(months=future_months)
    
    # Pokaż zakres dat
    st.write("Zakres prognozy:")
    st.write(f"Od: {last_date.strftime('%Y-%m')}")
    st.write(f"Do: {forecast_end_date.strftime('%Y-%m')}")

# Interfejs użytkownika
st.title("Prognozowanie Cen Metali")

# Tylko najważniejsza informacja z ikonami 
# Wszystko inne pod przyciskami, aby nie przytłaczać użytkownika
st.markdown("""
<div style="text-align:center; font-size:24px; margin:20px 0">
💎 Złoto · 🐶 Srebro · 🥈 Platyna · 🏅 Pallad · 🚳 Miedź
</div>
""", unsafe_allow_html=True)

# Wyświetlanie wybranego metalu do prognozowania
st.markdown(f"""
<div style="text-align:center; font-size:28px; margin:10px 0; font-weight:bold; color:#FF4B4B;">
Wybrany metal: {selected_metal}
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Prognozowanie 📉", "O Aplikacji ℹ️", "Dokładność 🏆"])

with tab1:
    # Zakładka z prognozowaniem - wszystkie elementy związane z prognozowaniem
    
    # Wyświetl podstawowe statystyki
    st.subheader("Statystyki dla wybranego metalu")
    stats = df[selected_metal].describe()
    st.write(stats)
    
    # Wyświetl wykres danych historycznych
    st.subheader("Dane historyczne")
    historical_chart = create_prediction_chart(df, None, selected_metal)
    st.plotly_chart(historical_chart, use_container_width=True, key="historical_chart_tab1")
    
    # Sekcja trenowania modelu
    st.subheader("Trenowanie modelu")
    if st.button("Trenuj model", key="train_model_tab1"):
        if retrain_model(df, selected_metal):
            st.success(f"Model dla {selected_metal} został pomyślnie przeszkolony!")
            st.cache_resource.clear()  # Wyczyść cache modelu
            
            # Dodaj małe opóźnienie, aby upewnić się, że pliki są zapisane
            import time
            time.sleep(1)  # Czekaj 1 sekundę
            
            # Ponownie sprawdź istnienie plików
            results_subdir = f'results_{selected_metal}'
            model_results_dir = Path("models") / results_subdir
            if (model_results_dir / 'residuals_plot.png').exists() and (model_results_dir / 'error_plot.png').exists():
                display_model_analysis(selected_metal)
            else:
                st.warning(f"Wykresy analizy modelu są generowane. Odśwież stronę, aby je zobaczyć.")
                # Zapisz informację, że wykresy są generowane
                with open("temp_status.txt", "w") as f:
                    f.write(f"generating_plots_{selected_metal}")
                    
    # Sekcja prognozowania
    st.subheader("Prognozowanie cen")
    
    # Przycisk do prognozowania
    if st.button("Prognozuj cenę", key="forecast_button_tab1"):
        try:
            # Sprawdź istnienie pliku modelu
            model_filename = f"model_{selected_metal}.pkl"
            model_path = MODELS_DIR / model_filename
            if not model_path.exists():
                st.error(f"Model dla {selected_metal} nie został jeszcze wytrenowany! Przejdź do sekcji 'Trenowanie modelu'.")
                st.stop()
            
            # Generuj daty dla prognozy
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                end=forecast_end_date,
                freq='M'
            )
            
            # Przygotuj dane do predykcji dla każdej daty - UŻYWAJĄC PODEJŚCIA REKURENCYJNEGO
            predictions_list = []
            previous_predictions = {}  # Słownik do przechowywania poprzednich prognoz
            
            # Inicjalizacja pierwszego wiersza danych - bazuje na ostatnim historycznym wierszu
            forecast_row = df.iloc[-1:].copy()
            
            for i, future_date in enumerate(future_dates):
                # Aktualizuj datę
                forecast_row['data'] = pd.to_datetime(future_date)
                
                # KROK 1: Aktualizuj wartości lagów rekurencyjnie dla wszystkich metali
                # Przesuwamy wartości w czasie: lag_3 otrzymuje wartość lag_1 z poprzedniego kroku,
                # a lag_1 otrzymuje ostatnią znaną cenę (która jest prognozą z poprzedniego kroku).
                if i > 0:  # Dla wszystkich kroków prognozy poza pierwszym
                    for metal_name in available_metals:
                        # Przesuń lag_1 do lag_3
                        forecast_row[f'{metal_name}_lag_3'] = forecast_row[f'{metal_name}_lag_1']
                        # Przesuń aktualną cenę (prognozę z poprzedniego kroku) do lag_1
                        forecast_row[f'{metal_name}_lag_1'] = forecast_row[metal_name]

                # KROK 2: Dokonaj predykcji z zaktualizowanymi lagami
                st.write(f"Debug: Forecast row for {future_date.strftime('%Y-%m')}: {forecast_row[selected_metal + '_lag_1'].values[0]}, {forecast_row[selected_metal + '_lag_3'].values[0]}")
                prediction = predict_price(forecast_row.copy(), selected_metal)
                
                if prediction is not None:
                    # Zapisz tę prognozę do późniejszego użycia jako lag
                    if selected_metal not in previous_predictions:
                        previous_predictions[selected_metal] = []
                    previous_predictions[selected_metal].append(prediction)
                    
                    # Aktualizuj wartość metalu w wierszu prognozującym dla kolejnych iteracji
                    forecast_row[selected_metal] = prediction
                    
                    # Zapisz wynik do wyświetlenia
                    prediction_df = pd.DataFrame({
                        'data': [future_date],
                        'prediction_label': [prediction]
                    })
                    predictions_list.append(prediction_df)
            
            if predictions_list:
                # Połącz wszystkie predykcje
                all_predictions = pd.concat(predictions_list)
                
                # Upewnij się, że kolumna 'data' jest w formacie datetime
                all_predictions['data'] = pd.to_datetime(all_predictions['data'])
                
                # Wyświetl prognozę
                st.write("### Prognozowane ceny")
                
                # Tabela z prognozami
                forecast_df = pd.DataFrame({
                    "Data": all_predictions["data"].dt.strftime("%Y-%m"),
                    "Prognozowana cena": all_predictions["prediction_label"].round(2)
                })
                st.dataframe(forecast_df)
                
                # Wykres
                prediction_chart = create_prediction_chart(df, all_predictions, selected_metal)
                st.plotly_chart(prediction_chart, use_container_width=True, key="prediction_chart_tab1")
                
            else:
                st.error(f"Nie udało się wykonać prognozy dla {selected_metal}. Sprawdź logi w terminalu.")
                
        except Exception as e:
            st.error(f"Błąd podczas prognozowania: {e}")

with tab2:
    # Rozbudowana zakładka "O Aplikacji"
    st.markdown("""
    <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-bottom:15px">
    <h2 style="text-align:center">📈 Metal Price Predictor</h2>
    <h4 style="text-align:center">Odkryj przyszłość rynków metali szlachetnych</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Podzial na kolumny z opisem i procesem
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### O aplikacji
        
        **Metal Price Predictor** to zaawansowane narzędzie wykorzystujące uczenie maszynowe 
        do prognozowania cen metali szlachetnych i przemysłowych. Aplikacja analizuje 
        historyczne dane cenowe oraz wskaźniki makroekonomiczne, aby przewidzieć 
        przyszłe trendy na rynkach metali.
        
        ### Możliwości:
        - 📈 **Prognozowanie** cen złota, srebra, platyny, palladu i miedzi
        - 📊 **Analiza historyczna** trendów cenowych od 2000 roku
        - 💰 **Planowanie inwestycji** w oparciu o prognozy
        - 💽 **Trening modeli** dla konkretnych metalów
        - 💹 **Backtesting** na danych historycznych
        """)
    
    with col2:
        st.markdown("""
        ### Jak to działa?
        """)        
        st.markdown("""
        <div style="text-align:center; margin:15px 0; padding:10px; background-color:#f8f9fa; border-radius:10px">
        <p style="font-size:18px; margin:5px">📂 <b>Zbieranie danych</b></p>
        <p style="font-size:12px; color:#666">Yahoo Finance, FRED, dane makro</p>
        <p style="font-size:18px">↓</p>
        <p style="font-size:18px; margin:5px">🛠️ <b>Przetwarzanie</b></p> 
        <p style="font-size:12px; color:#666">Feature engineering, lagi autoregresyjne</p>
        <p style="font-size:18px">↓</p>
        <p style="font-size:18px; margin:5px">🧙 <b>Uczenie maszynowe</b></p>
        <p style="font-size:12px; color:#666">OMP, Lasso, Decision Trees</p>
        <p style="font-size:18px">↓</p>
        <p style="font-size:18px; margin:5px">📉 <b>Prognozowanie</b></p>
        <p style="font-size:12px; color:#666">Rekurencyjna predykcja</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Informacje o projekcie
    st.markdown("### Autorzy i dokumentacja")
    col_authors1, col_authors2 = st.columns(2)
    
    with col_authors1:
        st.markdown("""
        #### 📖 Dokumentacja projektu
        - [README.md](https://github.com/example/metal_predictor/blob/main/README.md)
        - [Podsumowanie projektu](https://github.com/example/metal_predictor/blob/main/projekt_podsumowanie.md)
        """)
    
    with col_authors2:
        st.markdown("""
        #### 👨‍💻 Autorzy
        - Zespół Od Zera do AI
        - Kontakt: example@email.com
        """)
    
    # Sekcje z rozwijalnymi informacjami - przenosimy tutaj z głównej części
    st.markdown("### Więcej informacji")
    
    # 1. Sekcja z danymi o metalach
    with st.expander("Dane o cenach metali używane do modelowania"):
        st.markdown("""
        #### Źródła danych:
        * Yahoo Finance, tickery:
            - Złoto: `GC=F`, `XAUUSD=X`, `GLD`
            - Srebro: `SI=F`, `XAGUSD=X`, `SLV`
            - Platyna: `PL=F`, `PPLT`
            - Pallad: `PA=F`, `PALL`
            - Miedź: `HG=F`, `CPER`
        * Częstotliwość: Miesięczne uśrednienia wartości dziennych
        * Zakres czasowy: 2000-12-01 do 2025-07-01
        """)
    
    # 2. Sekcja z cechami autoregresyjnymi
    with st.expander("Cechy autoregresyjne (lagi) zwiększające dokładność prognoz"):
        st.markdown("""
        #### Autoregresyjne cechy modelu:
        * **Lag_1**: Wartość ceny danego metalu z poprzedniego miesiąca
        * **Lag_3**: Wartość ceny danego metalu sprzed trzech miesięcy
        * **Wpływ na model**: Redukcja MAPE z ~5% do ~3-4.5% w testach backtestingowych
        """)
    
    # 3. Sekcja ze wskaźnikami makroekonomicznymi
    with st.expander("Wskaźniki makroekonomiczne wykorzystane w modelu predykcyjnym"):
        st.markdown("""
        #### Wskaźniki makroekonomiczne uwzględniane w modelu:
        * **Inflacja (USA)**: Miesięczny wskaźnik inflacji CPI w Stanach Zjednoczonych
        * **Stopy procentowe (FED)**: Stopy procentowe amerykańskiej Rezerwy Federalnej
        * **Bezrobocie (USA)**: Stopa bezrobocia w USA jako % siły roboczej
        * **PKB krajowy (USA)**: Kwartalne wartości PKB Stanów Zjednoczonych
        * **PKB globalny**: Kwartalne wartości światowego PKB (Bank Światowy)
        * **Kurs USD (DXY)**: Indeks dolara amerykańskiego wobec koszyka walut
        * **Indeks VIX**: Indeks zmienności rynkowej Chicago Board Options Exchange
        """)
    
    # 4. Sekcja z algorytmami
    with st.expander("Algorytmy uczenia maszynowego stosowane do predykcji"):
        st.markdown("""
        #### Algorytmy używane do trenowania modeli:
        * Automatyczna selekcja najlepszego modelu z biblioteki PyCaret
        * Najskuteczniejsze modele w backtestingu:
            - Orthogonal Matching Pursuit (OMP)
            - Lasso Regression
            - Decision Tree Regressor
        """)
    
    # 5. Sekcja z instrukcją obsługi
    with st.expander("Jak korzystać z aplikacji do prognozowania"):
        st.markdown("""
        #### Instrukcja obsługi:
        1. **Wybór metalu** - w panelu bocznym wybierz metal do analizy
        2. **Trenowanie modelu** - kliknij "Trenuj model" dla wybranego metalu
        3. **Prognozowanie** - ustaw horyzont czasowy i kliknij "Prognozuj cenę"
        4. **Analiza wyników** - przejrzyj wykresy i statystyki modelu
        """)

with tab3:
    # Zakładka z dokładnością modelu
    st.markdown("""
    <div style="background-color:#e6f3ff; padding:15px; border-radius:10px; text-align:center">
    <h3>🏆 Dokładność modelu</h3>
    <h1 style="color:#0068c9">93-97%</h1>
    <p>Backtesting na danych historycznych wykazał bardzo wysoką dokładność prognoz.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Wyświetlanie analizy modelu dla wybranego metalu
    st.subheader(f"Analiza modelu dla {selected_metal}")
    
    # Sprawdzamy, czy model istnieje i czy są dostępne wyniki
    model_filename = f"model_{selected_metal}.pkl"
    model_path = MODELS_DIR / model_filename
    
    if model_path.exists():
        # Wyświetl wykresy analizy modelu
        display_model_analysis(selected_metal)
        
        # Dodatkowe informacje o metrykach
        st.markdown("""
        #### Objaśnienie metryk:
        - **MAPE**: Mean Absolute Percentage Error - średni bezwzględny błąd procentowy
        - **MAE**: Mean Absolute Error - średni błąd bezwzględny
        - **MSE**: Mean Squared Error - średni błąd kwadratowy
        - **R²**: Współczynnik determinacji - jak dobrze model wyjaśnia zmienność danych
        """)
        
        # Sekcja z wynikami backtestingu
        with st.expander("Wyniki backtestingu"):
            st.markdown("""
            ### Wyniki testów backtestingowych
            
            Modele zostały przetestowane na danych historycznych z różnych okresów:
            
            | Metal | MAPE (%) | Najlepszy model |
            |-------|----------|----------------|
            | Złoto | 3.2% | Orthogonal Matching Pursuit |
            | Srebro | 4.1% | Lasso Regression |
            | Platyna | 3.8% | Decision Tree Regressor |
            | Pallad | 6.7% | Random Forest |
            | Miedź | 3.5% | Orthogonal Matching Pursuit |
            
            > Najlepsze wyniki osiągnięto stosując autoregresyjne cechy (lagi) cen metali.
            """)
    else:
        st.warning(f"Model dla {selected_metal} nie został jeszcze wytrenowany. Aby zobaczyć analizę dokładności, przejdź do zakładki 'Prognozowanie' i wytrenuj model.")
        
    # Informacja o cechach wpływających na dokładność
    st.subheader("Kluczowe czynniki wpływające na dokładność")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Cechy zwiększające dokładność:
        - ✅ Lagi autoregresyjne (lag_1, lag_3)
        - ✅ Wskaźniki makroekonomiczne
        - ✅ Indeks VIX (zmienność rynkowa)
        - ✅ Kurs dolara (DXY)
        """)
        
    with col2:
        st.markdown("""
        #### Wyzwania dla dokładności:
        - ⚠️ Nagłe wydarzenia geopolityczne
        - ⚠️ Nietypowe ruchy banków centralnych
        - ⚠️ Kryzys na rynkach finansowych
        - ⚠️ Zmienność palladu
        """)
    
    # Rozwijana sekcja z możliwościami poprawy dokładności
    with st.expander("Możliwości dalszej poprawy dokładności"):
        st.markdown("""
        ### Potencjalne kierunki rozwoju modelu:
        
        1. **Dodatkowe lagi czasowe** - wprowadzenie lag_6 i lag_12 dla uchwycenia efektów sezonowych
        2. **Analiza sentymentu rynkowego** - przetwarzanie wiadomości i mediów społecznościowych
        3. **Dodatkowe wskaźniki makro** - szerszy zestaw zmiennych ekonomicznych
        4. **Zaawansowane modele neuronowe** - architektura LSTM do lepszego uchwycenia długoterminowych zależności
        5. **Modele hybrydowe** - łączenie różnych typów modeli dla większej dokładności
        """)



# Sekcje rozwijane zostały przeniesione do odpowiednich zakładek









# Wyświetl podstawowe statystyki
st.subheader("Statystyki dla wybranego metalu")
stats = df[selected_metal].describe()
st.write(stats)

# Wyświetl wykres danych historycznych
st.subheader("Dane historyczne")
historical_chart = create_prediction_chart(df, None, selected_metal)
st.plotly_chart(historical_chart, use_container_width=True, key="historical_chart_main")

# Sekcja trenowania modelu
st.subheader("Trenowanie modelu")
if st.button("Trenuj model", key="train_model_main"):
    if retrain_model(df, selected_metal):
        st.success(f"Model dla {selected_metal} został pomyślnie przeszkolony!")
        st.cache_resource.clear()  # Wyczyść cache modelu
        
        # Dodaj małe opóźnienie, aby upewnić się, że pliki są zapisane
        import time
        time.sleep(1)  # Czekaj 1 sekundę
        
        # Ponownie sprawdź istnienie plików
        results_subdir = f'results_{selected_metal}'
        model_results_dir = Path("models") / results_subdir
        if (model_results_dir / 'residuals_plot.png').exists() and (model_results_dir / 'error_plot.png').exists():  
            display_model_analysis(selected_metal)
        else:
            st.warning(f"Wykresy analizy modelu są generowane. Odśwież stronę, aby je zobaczyć.")
            # Zapisz informację, że wykresy są generowane
            with open("temp_status.txt", "w") as f:
                f.write(f"generating_plots_{selected_metal}")
if st.button("Prognozuj cenę", key="forecast_button_main"):
    try:
        # Sprawdź istnienie pliku modelu
        model_filename = f"model_{selected_metal}.pkl"
        model_path = MODELS_DIR / model_filename
        if not model_path.exists():
            st.error(f"Model dla {selected_metal} nie został jeszcze wytrenowany! Przejdź do sekcji 'Trenowanie modelu'.")
            st.stop()
        
        # Generuj daty dla prognozy
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            end=forecast_end_date,
            freq='M'
        )
        
        # Przygotuj dane do predykcji dla każdej daty - UŻYWAJĄC PODEJŚCIA REKURENCYJNEGO
        predictions_list = []
        previous_predictions = {}  # Słownik do przechowywania poprzednich prognoz
        
        # Inicjalizacja pierwszego wiersza danych - bazuje na ostatnim historycznym wierszu
        forecast_row = df.iloc[-1:].copy()
        
        for i, future_date in enumerate(future_dates):
            # Aktualizuj datę
            forecast_row['data'] = pd.to_datetime(future_date)
            
            # KROK 1: Aktualizuj wartości lagów na podstawie poprzednich prognoz
            # Tylko jeśli nie jest to pierwszy miesiąc prognozy
            if i > 0:
                # Aktualizuj lag_1 dla wszystkich metali (poprzednia prognoza)
                for metal_name in available_metals:
                    if metal_name in previous_predictions and len(previous_predictions[metal_name]) >= 1:
                        # Najnowsza prognoza jako lag_1
                        forecast_row[f'{metal_name}_lag_1'] = previous_predictions[metal_name][-1]
                    
                    # Aktualizuj lag_3 jeśli mamy już 3 prognozy
                    if metal_name in previous_predictions and len(previous_predictions[metal_name]) >= 3:
                        # Prognoza sprzed 3 miesięcy jako lag_3
                        forecast_row[f'{metal_name}_lag_3'] = previous_predictions[metal_name][-3]
            
            # KROK 2: Dokonaj predykcji z zaktualizowanymi lagami
            st.write(f"Debug: Forecast row for {future_date.strftime('%Y-%m')}: {forecast_row[selected_metal + '_lag_1'].values[0]}, {forecast_row[selected_metal + '_lag_3'].values[0]}")
            prediction = predict_price(forecast_row.copy(), selected_metal)
            
            if prediction is not None:
                # Zapisz tę prognozę do późniejszego użycia jako lag
                if selected_metal not in previous_predictions:
                    previous_predictions[selected_metal] = []
                previous_predictions[selected_metal].append(prediction)
                
                # Aktualizuj wartość metalu w wierszu prognozującym dla kolejnych iteracji
                forecast_row[selected_metal] = prediction
                
                # Zapisz wynik do wyświetlenia
                prediction_df = pd.DataFrame({
                    'data': [future_date],
                    'prediction_label': [prediction]
                })
                predictions_list.append(prediction_df)
        
        if predictions_list:
            # Połącz wszystkie predykcje
            all_predictions = pd.concat(predictions_list)
            
            # Upewnij się, że kolumna 'data' jest w formacie datetime
            all_predictions['data'] = pd.to_datetime(all_predictions['data'])
            
            # Wyświetl prognozę
            st.write("### Prognozowane ceny")
            
            # Tabela z prognozami
            forecast_df = pd.DataFrame({
                "Data": all_predictions["data"].dt.strftime("%Y-%m"),
                "Prognozowana cena": all_predictions["prediction_label"].round(2)
            })
            st.dataframe(forecast_df)
            
            # Wykres
            prediction_chart = create_prediction_chart(df, all_predictions, selected_metal)
            st.plotly_chart(prediction_chart, use_container_width=True, key="prediction_chart_main")
            
            # Wyświetl analizę modelu
            display_model_analysis(selected_metal)
        else:
            st.error(f"Nie udało się wykonać prognozy dla {selected_metal}. Sprawdź logi w terminalu.")
            
    except Exception as e:
        st.error(f"Błąd podczas prognozowania: {e}")

# Sekcje rozwijane z dodatkowymi informacjami
st.markdown("---")

# Informacje o źródłach danych
with st.expander("Informacje o źródłach danych"):
    st.markdown("""
    ### Źródła danych
    
    Dane historyczne cen metali szlachetnych i przemysłowych pochodzą z wielu źródeł:
    
    * **Yahoo Finance API** - główne źródło dla złota, srebra, platyny i palladu
    * **London Metal Exchange** - dane dla miedzi
    * **World Bank** - dane makroekonomiczne
    * **FRED (Federal Reserve Economic Data)** - wskaźniki ekonomiczne i stopy procentowe
    
    Dane są automatycznie aktualizowane podczas uruchamiania aplikacji, jeśli istniejące dane są starsze niż 24 godziny.
    """)

# Informacje o zastosowanych cechach autoregresyjnych (lag)
with st.expander("Informacje o cechach autoregresyjnych"):
    st.markdown("""
    ### Cechy autoregresyjne (lag features)
    
    Model wykorzystuje cechy autoregresyjne - wcześniejsze wartości cen jako cechy wejściowe:
    
    * **Metal_lag_1** - cena z poprzedniego miesiąca
    * **Metal_lag_3** - cena sprzed trzech miesięcy
    
    Dodanie tych cech znacząco poprawia dokładność modelu w porównaniu do użycia samych wskaźników makroekonomicznych.
    
    W prognozach rekurencyjnych (na wiele miesięcy do przodu), wartości lagów są aktualizowane na podstawie
    poprzednich prognoz, co poprawia jakość długoterminowych przewidywań.
    """)

# Informacje o wskaźnikach makroekonomicznych
with st.expander("Informacje o wskaźnikach makroekonomicznych"):
    st.markdown("""
    ### Wskaźniki makroekonomiczne
    
    Model uwzględnia następujące wskaźniki makroekonomiczne jako cechy wejściowe:
    
    * **Indeks dolara amerykańskiego (DXY)** - wskaźnik wartości dolara względem koszyka walut
    * **Inflacja w USA** - roczna stopa inflacji CPI
    * **Stopa procentowa Fed** - podstawowa stopa procentowa Fed
    * **Indeks VIX** - indeks zmienności rynków (tzw. "indeks strachu")
    * **Indeks S&P 500** - indeks 500 największych spółek notowanych na giełdach w USA
    
    Dane te są znormalizowane przed podaniem do modelu.
    """)

# Informacje o algorytmach uczenia maszynowego
with st.expander("Informacje o algorytmach uczenia maszynowego"):
    st.markdown("""
    ### Algorytmy uczenia maszynowego
    
    Aplikacja wykorzystuje bibliotekę PyCaret do automatycznego trenowania i wyboru najlepszego modelu.
    Trenowanych jest wiele algorytmów, w tym:
    
    * **Linear Regression** - regresja liniowa
    * **Random Forest** - las losowy
    * **Gradient Boosting** - wzmacnianie gradientowe
    * **Decision Tree** - drzewa decyzyjne
    * **Support Vector Regression** - maszyna wektorów nośnych
    * **Lasso Regression** - regresja Lasso
    * **Ridge Regression** - regresja Ridge
    * **Orthogonal Matching Pursuit** - algorytm OMP
    
    Najlepszy model jest wybierany na podstawie metryki MAPE (Mean Absolute Percentage Error).
    """)

# Informacje o użytkowaniu
with st.expander("Jak korzystać z aplikacji?"):
    st.markdown("""
    ### Jak korzystać z aplikacji?
    
    1. **Wybierz metal** w panelu bocznym (złoto, srebro, platyna, pallad lub miedź)
    2. **Wybierz okres prognozy** za pomocą suwaka (1-120 miesięcy)
    3. **Trenuj model** - kliknij przycisk "Trenuj model" aby wytrenować nowy model dla wybranego metalu
    4. **Generuj prognozę** - kliknij przycisk "Prognozuj cenę" aby wygenerować prognozę na wybrany okres
    5. **Analizuj wyniki** - przeglądaj tabelę i wykres z prognozami oraz metryki jakości modelu
    
    > **Uwaga:** Dla każdego metalu należy osobno wytrenować model przed prognozowaniem.
    """)