import streamlit as st
import pandas as pd
from typing import List, Dict, Optional
from src.config import AVAILABLE_METALS, DEFAULT_START_DATE, DEFAULT_END_DATE
from src.data_loader import get_all_data
from src.processing import load_combined_data
from src.model_training import train_model
from src.predict import predict_price

@st.cache_data(ttl=3600)
def load_and_validate_data():
    """Ładuje i waliduje dane, aktualizując je w razie potrzeby."""
    try:
        df = load_combined_data()
        if df is None or df.empty:
            st.info("Brak danych, pobieranie...")
            get_all_data(DEFAULT_START_DATE, DEFAULT_END_DATE)
            df = load_combined_data()
            if df is None or df.empty:
                st.error("Nie udało się załadować danych.")
                return None
        
        required_columns = ["data"] + AVAILABLE_METALS
        if not all(col in df.columns for col in required_columns):
            st.error("Dane nie zawierają wymaganych kolumn.")
            return None
            
        return df
    except Exception as e:
        st.error(f"Błąd podczas ładowania danych: {e}")
        return None

def handle_training(df, target_metal):
    """Obsługuje proces trenowania modelu."""
    try:
        with st.spinner(f'Trenuję model dla {target_metal}...'):
            train_model(df, target_metal)
            st.success(f"Model dla {target_metal} został pomyślnie wytrenowany!")
            st.cache_resource.clear() # Czyści cache, aby odświeżyć widok wyników
    except Exception as e:
        st.error(f"Błąd podczas trenowania modelu: {e}")

def handle_forecasting(df, selected_metal, last_date, forecast_end_date):
    """Obsługuje proces generowania i wyświetlania prognozy."""
    try:
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            end=forecast_end_date,
            freq='M'
        )
        
        predictions_list = []
        forecast_row = df.iloc[-1:].copy()

        with st.spinner("Generowanie prognozy..."):
            for i, future_date in enumerate(future_dates):
                forecast_row['data'] = pd.to_datetime(future_date)
                
                if i > 0:
                    for metal_name in AVAILABLE_METALS:
                        forecast_row[f'{metal_name}_lag_3'] = forecast_row[f'{metal_name}_lag_1']
                        forecast_row[f'{metal_name}_lag_1'] = forecast_row[metal_name]

                prediction = predict_price(forecast_row.copy(), selected_metal)
                if prediction is None:
                    st.error("Prognoza nie powiodła się. Sprawdź, czy model jest wytrenowany.")
                    return

                forecast_row[selected_metal] = prediction
                prediction_df = pd.DataFrame({'data': [future_date], 'prediction_label': [prediction]})
                predictions_list.append(prediction_df)
        
        if predictions_list:
            all_predictions = pd.concat(predictions_list)
            st.session_state['prediction_data'] = all_predictions
            st.success("Prognoza została wygenerowana!")
        else:
            st.error("Nie udało się wygenerować prognozy.")
            
    except Exception as e:
        st.error(f"Błąd podczas prognozowania: {e}")
