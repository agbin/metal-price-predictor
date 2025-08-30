import pandas as pd
from pycaret.regression import *
import json
from pathlib import Path

def train_with_pycaret():
    # Wczytaj dane z nowymi cechami
    df = pd.read_csv("../data/features/engineered_features.csv")
    df['data'] = pd.to_datetime(df['data'])
    
    # Inicjalizuj eksperyment
    print("Inicjalizacja eksperymentu...")
    exp = setup(
        data=df,
        target='Złoto',
        train_size=0.8,
        fold=5,
        fold_strategy='timeseries',  # Używamy walidacji krzyżowej dla szeregów czasowych
        numeric_features=[col for col in df.columns if col not in ['data', 'Złoto']],
        fold_shuffle=False,  # Nie tasujemy danych (to szereg czasowy)
        session_id=42,
        verbose=False
    )
    
    # Porównaj wszystkie modele
    print("\nPorównywanie modeli...")
    best_model = compare_models(
        sort='MAE',  # Sortuj po MAE
        n_select=1,  # Wybierz najlepszy model
        verbose=True
    )
    
    # Dostrój najlepszy model
    print("\nDostrajanie najlepszego modelu...")
    tuned_model = tune_model(
        best_model,
        optimize='MAE',
        n_iter=50,  # Liczba iteracji dla optymalizacji
        verbose=True
    )
    
    # Finalizuj model
    final_model = finalize_model(tuned_model)
    
    # Zapisz model
    print("\nZapisywanie modelu...")
    models_dir = Path("models/pycaret")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Zapisz model
    save_model(final_model, str(models_dir / 'best_model'))
    
    # Pobierz wyniki
    results = pull()
    
    # Zapisz wyniki do JSON
    results_dict = results.to_dict('records')
    with open(models_dir / "model_comparison_results.json", 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"\nZapisano model i wyniki w: {models_dir}")
    print("\nNajlepsze modele według MAE:")
    print(results.head().to_string())

if __name__ == "__main__":
    train_with_pycaret()
