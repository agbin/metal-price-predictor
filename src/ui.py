import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from src.config import AVAILABLE_METALS, RESULTS_DIR
from typing import Dict, Optional
import os

def create_sidebar(df):
    """Tworzy i zarządza panelem bocznym aplikacji."""
    with st.sidebar:
        st.header("Ustawienia")
        selected_metal = st.selectbox("Wybierz metal", AVAILABLE_METALS, key="metal_selector")
        
        st.markdown("---")
        
        last_date = pd.to_datetime(df['data']).max()
        forecast_months = st.slider(
            "Wybierz okres prognozy (w miesiącach)", 
            min_value=1, 
            max_value=120, 
            value=12, 
            step=1,
            key="forecast_slider"
        )
        
        forecast_end_date = last_date + pd.DateOffset(months=forecast_months)
        st.info(f"Prognoza do: {forecast_end_date.strftime('%B %Y')}")
        
    return selected_metal, forecast_months, last_date, forecast_end_date

def create_main_view(selected_metal):
    """Tworzy główny widok aplikacji z tytułem i zakładkami."""
    st.markdown(f"""
    <div style="text-align:center; font-size:28px; margin:10px 0; font-weight:bold; color:#FF4B4B;">
    Wybrany metal: {selected_metal}
    </div>
    """, unsafe_allow_html=True)
    
    return st.tabs(["Prognozowanie 📉", "O Aplikacji ℹ️", "Dokładność 🏆"])

def get_current_model_info(metal: str, model_dir: str = 'models') -> Optional[str]:
    """Odczytuje nazwę zapisanego modelu z pliku tekstowego."""
    model_name_path = os.path.join(model_dir, f'model_{metal}_name.txt')
    if os.path.exists(model_name_path):
        with open(model_name_path, 'r') as f:
            return f.read().strip()
    return None

def display_prediction_tab(df, selected_metal, train_callback, forecast_callback):
    """Wyświetla zawartość zakładki 'Prognozowanie'."""
    st.subheader("Statystyki dla wybranego metalu")
    st.write(df[selected_metal].describe())

    st.subheader("Wykres historyczny")
    historical_chart = create_price_chart(df, None, selected_metal)
    st.plotly_chart(historical_chart, use_container_width=True)

    st.subheader("Trenowanie modelu")
    col1, col2 = st.columns(2)
    with col1:
        current_model_name = get_current_model_info(selected_metal)
        if current_model_name:
            st.info(f"**Aktualnie używany model:** `{current_model_name}`\n\nZostał on automatycznie wybrany jako najdokładniejszy na podstawie danych historycznych.")
        else:
            st.warning("Brak wytrenowanego modelu. Kliknij przycisk poniżej, aby go przygotować.")

        st.button("Trenuj model", on_click=train_callback, help="Kliknij, aby ponownie wytrenować model na najnowszych danych. Może to zająć kilka minut.")
        st.caption("Tworzy i zapisuje nowy model prognostyczny na podstawie całego zbioru danych.")

    with col2:
        st.button("Prognozuj", on_click=forecast_callback, help="Kliknij, aby wygenerować prognozę cen na wybrany okres.")
        st.caption("Używa istniejącego modelu do szybkiego wygenerowania przyszłych cen.")

def display_about_tab():
    """Wyświetla zawartość zakładki 'O Aplikacji'."""
    st.header("Informacje o Aplikacji")
    st.markdown("""
    Ta aplikacja została stworzona w celu demonstracji możliwości prognozowania szeregów czasowych 
    z wykorzystaniem biblioteki PyCaret.
    """)

    st.subheader("Na podstawie jakich danych działa model?")
    st.markdown("""
    Model prognostyczny bierze pod uwagę szereg czynników, aby jak najdokładniej przewidzieć przyszłe ceny metali. Poniżej znajduje się kompletna lista danych wejściowych:

    *   **Ceny historyczne (Lagi):** Wartości cen metalu z poprzednich dni (np. `Złoto_lag_1`).
    *   **Wzajemne zależności rynkowe:** Ceny pozostałych metali (np. cena Srebra jest brana pod uwagę przy prognozowaniu ceny Złota).
    *   **Kluczowe wskaźniki makroekonomiczne:**
        *   `inflacja` - Poziom inflacji
        *   `stopy_procentowe` - Wysokość stóp procentowych
        *   `bezrobocie` - Stopa bezrobocia
        *   `pkb` - Produkt Krajowy Brutto
        *   `pkb_global` - Globalny Produkt Krajowy Brutto
    *   **Wskaźniki rynków finansowych:**
        *   `kurs_usd` - Kurs dolara amerykańskiego
        *   `indeks_vix` - Indeks zmienności rynkowej, tzw. "indeks strachu"
    *   **Sentyment z globalnych wiadomości:**
        *   `cb_sentiment_index` - Indeks sentymentu obliczony na podstawie analizy globalnych wiadomości.
    """)
    with st.expander("Źródła danych"):
        st.markdown("""
        - **Yahoo Finance API**
        - **London Metal Exchange**
        - **World Bank**
        - **FRED (Federal Reserve Economic Data)**
        """)
    with st.expander("Wykorzystane technologie"):
        st.markdown("""
        - **Streamlit** - interfejs użytkownika
        - **PyCaret** - automatyzacja uczenia maszynowego
        - **Pandas** - manipulacja danymi
        - **Plotly** - interaktywne wykresy
        """)

def display_accuracy_tab(selected_metal):
    """Wyświetla zawartość zakładki 'Dokładność' z wynikami modelu."""
    st.header(f"Dokładność modelu dla: {selected_metal}")

    # Ścieżka do pliku z metrykami
    metrics_path = os.path.join('results', f'results_{selected_metal}', 'model_metrics.csv')

    if os.path.exists(metrics_path):
        st.subheader("Metryki wydajności modelu")
        st.markdown("""
        Poniższa tabela przedstawia kluczowe wskaźniki oceniające dokładność ostatnio wytrenowanego modelu.
        Metryki te zostały obliczone na zbiorze testowym, który nie był używany podczas trenowania.
        """)
        try:
            metrics_df = pd.read_csv(metrics_path)
                        # Zastosuj formatowanie tylko do kolumn numerycznych
            numeric_cols = metrics_df.select_dtypes(include='number').columns
            st.dataframe(metrics_df.style.format('{:.4f}', subset=numeric_cols))

            with st.expander("Objaśnienie metryk"):
                st.markdown("""
                - **MAE (Mean Absolute Error)**: Średni bezwzględny błąd prognozy. Im niższa wartość, tym lepiej.
                - **MSE (Mean Squared Error)**: Średni kwadratowy błąd. Kara za duże błędy jest większa. Im niższa wartość, tym lepiej.
                - **RMSE (Root Mean Squared Error)**: Pierwiastek z MSE. Jest w tej samej jednostce co prognozowana cena. Im niższa wartość, tym lepiej.
                - **R2 (R-squared)**: Współczynnik determinacji. Mówi, jaki procent zmienności ceny jest wyjaśniany przez model. Wartości bliższe 1 są lepsze.
                - **RMSLE (Root Mean Squared Log Error)**: Podobne do RMSE, ale liczone na logarytmach. Mniej wrażliwe na duże błędy w prognozach wysokich cen.
                - **MAPE (Mean Absolute Percentage Error)**: Średni procentowy błąd. Pokazuje, o ile procent średnio myli się model.
                """)
        except Exception as e:
            st.error(f"Nie udało się wczytać pliku z metrykami: {e}")
    else:
        st.info("Pamiętaj, że najpierw musisz wytrenować model dla wybranego metalu, aby metryki i wykresy mogły się pojawić.")

    # Usunięto wyświetlanie wykresów analizy modelu na życzenie użytkownika.

def create_price_chart(historical_data, prediction_data, metal_name):
    """Tworzy interaktywny wykres cen z danymi historycznymi i prognozą."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_data['data'], 
        y=historical_data[metal_name], 
        mode='lines', 
        name='Dane historyczne'
    ))
    if prediction_data is not None:
        fig.add_trace(go.Scatter(
            x=prediction_data['data'], 
            y=prediction_data['prediction_label'], 
            mode='lines', 
            name='Prognoza', 
            line=dict(color='red', dash='dash')
        ))
    fig.update_layout(
        title=f'Ceny historyczne i prognoza dla: {metal_name}',
        xaxis_title='Data',
        yaxis_title='Cena (USD)',
        template='plotly_dark'
    )
    return fig
