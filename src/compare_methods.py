import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Wczytaj dane
df = pd.read_csv('../data/processed/combined_real_and_generated_data.csv')
df['data'] = pd.to_datetime(df['data'])

# Wybierz dane z lat 2000-2004 dla złota
mask = (df['data'].dt.year >= 2000) & (df['data'].dt.year <= 2004)
df_subset = df[mask].copy()

# Zachowaj oryginalne wartości złota
original_gold = df_subset['Złoto'].copy()

# 1. Interpolacja liniowa
df_subset['Złoto_interpolacja'] = df_subset['Złoto'].interpolate(method='linear')

# 2. Regresja liniowa
def fill_with_linear_regression(df, target_col):
    df = df.copy()
    # Użyj tylko kolumn numerycznych jako cechy
    feature_cols = ['inflacja', 'stopy_procentowe', 'bezrobocie', 'kurs_usd']
    
    # Przygotuj dane treningowe (wiersze bez braków)
    train_mask = df[target_col].notna()
    X_train = df[train_mask][feature_cols]
    y_train = df[train_mask][target_col]
    
    # Trenuj model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Uzupełnij braki
    missing_mask = df[target_col].isna()
    X_missing = df[missing_mask][feature_cols]
    df.loc[missing_mask, target_col + '_regresja'] = model.predict(X_missing)
    
    # Skopiuj istniejące wartości
    df.loc[~missing_mask, target_col + '_regresja'] = df.loc[~missing_mask, target_col]
    return df

# 3. KNN
def fill_with_knn(df, target_col, n_neighbors=5):
    df = df.copy()
    # Użyj tylko kolumn numerycznych jako cechy
    feature_cols = ['inflacja', 'stopy_procentowe', 'bezrobocie', 'kurs_usd']
    
    # Przygotuj dane
    X = df[feature_cols + [target_col]].copy()
    
    # Uzupełnij braki używając KNN
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_imputed = imputer.fit_transform(X)
    
    # Zapisz wyniki
    df[target_col + '_knn'] = X_imputed[:, -1]
    return df

# Zastosuj metody
df_subset = fill_with_linear_regression(df_subset, 'Złoto')
df_subset = fill_with_knn(df_subset, 'Złoto')

# Wyświetl wyniki dla wierszy z brakującymi danymi
missing_mask = original_gold.isna()
comparison = df_subset[missing_mask][['data', 'Złoto_interpolacja', 'Złoto_regresja', 'Złoto_knn']]
comparison = comparison.round(2)

print("\nPorównanie metod uzupełniania braków w danych:")
print("\nData           Interpolacja  Regresja    KNN")
print("-" * 50)
for _, row in comparison.iterrows():
    print(f"{row['data'].strftime('%Y-%m'):10} {row['Złoto_interpolacja']:12.2f} {row['Złoto_regresja']:10.2f} {row['Złoto_knn']:10.2f}")

# Zapisz wyniki do pliku CSV
comparison.to_csv('../data/processed/gold_comparison.csv', index=False)

# Wykres
plt.figure(figsize=(15, 7))
plt.plot(df_subset['data'], df_subset['Złoto_interpolacja'], 'b-', label='Interpolacja liniowa')
plt.plot(df_subset['data'], df_subset['Złoto_regresja'], 'r-', label='Regresja liniowa')
plt.plot(df_subset['data'], df_subset['Złoto_knn'], 'g-', label='KNN')
plt.plot(df_subset['data'], original_gold, 'k.', label='Oryginalne dane')

plt.title('Porównanie metod uzupełniania braków w danych dla złota (2000-2004)')
plt.xlabel('Data')
plt.ylabel('Cena złota')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../results/gold_comparison.png')
