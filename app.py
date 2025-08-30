import streamlit as st
from src.app_logic import load_and_validate_data, handle_training, handle_forecasting
from src.ui import (
    create_sidebar, 
    create_main_view, 
    display_prediction_tab, 
    display_about_tab, 
    display_accuracy_tab,
    create_price_chart
)

st.set_page_config(layout="wide", page_title="Metal Price Predictor")

def main():
    """Główna funkcja aplikacji Streamlit."""
    st.title("📈 Metal Price Predictor")

    # Ładowanie danych
    df = load_and_validate_data()
    if df is None:
        st.stop()

    start_date = df['data'].min().strftime('%Y-%m-%d')
    end_date = df['data'].max().strftime('%Y-%m-%d')

    st.info(
        f""" 
        **Informacje o Aplikacji:**
        - **Cel:** Prognozowanie cen metali szlachetnych i przemysłowych.
        - **Źródła danych:** Dane pochodzą z wielu źródeł, m.in. **Yahoo Finance**, **FRED**, **World Bank**, **LME** oraz **Projekt GDELT**.
        - **Baza dla prognoz:** Model korzysta z danych historycznych z okresu od **{start_date}** do **{end_date}**.
        """
    )

    # Inicjalizacja stanu sesji dla prognozy
    if 'prediction_data' not in st.session_state:
        st.session_state['prediction_data'] = None

    # Tworzenie interfejsu
    selected_metal, forecast_months, last_date, forecast_end_date = create_sidebar(df)
    tab1, tab2, tab3 = create_main_view(selected_metal)

    # Definiowanie akcji (callbacków)
    def train_callback():
        handle_training(df, selected_metal)

    def forecast_callback():
        handle_forecasting(df, selected_metal, last_date, forecast_end_date)

    # Wyświetlanie zawartości zakładek
    with tab1:
        display_prediction_tab(df, selected_metal, train_callback, forecast_callback)
        # Jeśli prognoza istnieje w stanie sesji, wyświetl ją
        if st.session_state['prediction_data'] is not None:
            st.subheader("Wynik ostatniej prognozy")
            prediction_chart = create_price_chart(df, st.session_state['prediction_data'], selected_metal)
            st.plotly_chart(prediction_chart, use_container_width=True)
            st.dataframe(st.session_state['prediction_data'])

    with tab2:
        display_about_tab()

    with tab3:
        display_accuracy_tab(selected_metal)

if __name__ == "__main__":
    main()
