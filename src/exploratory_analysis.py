import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(filepath=None):
    """
    Wczytuje dane do analizy.
    
    Args:
        filepath (str, optional): Ścieżka do pliku z danymi. 
            Domyślnie używa processed_data_raw.csv
    
    Returns:
        pd.DataFrame: Wczytane dane
    """
    if filepath is None:
        filepath = Path("../data/processed/processed_data_raw.csv")
    return pd.read_csv(filepath, parse_dates=['data'])

def basic_statistics(df):
    """
    Generuje podstawowe statystyki dla wszystkich kolumn numerycznych.
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
        
    Returns:
        pd.DataFrame: Statystyki opisowe
    """
    return df.describe()

def plot_metal_prices(df, save_path=None):
    """
    Generuje wykres cen metali w czasie.
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
        save_path (str, optional): Ścieżka do zapisu wykresu
    """
    metals = ['Złoto', 'Srebro', 'Platyna', 'Pallad', 'Miedź']
    
    plt.figure(figsize=(15, 8))
    for metal in metals:
        plt.plot(df['data'], df[metal], label=metal)
    
    plt.title('Ceny metali w czasie')
    plt.xlabel('Data')
    plt.ylabel('Cena')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def correlation_heatmap(df, save_path=None):
    """
    Generuje mapę cieplną korelacji między zmiennymi.
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
        save_path (str, optional): Ścieżka do zapisu wykresu
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Mapa korelacji między zmiennymi')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    # Utworzenie katalogu na wykresy
    output_dir = Path("../results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Wczytanie danych
    df = load_data()
    
    # Podstawowe statystyki
    stats = basic_statistics(df)
    print("\nPodstawowe statystyki:")
    print(stats)
    
    # Wykresy
    plot_metal_prices(df, output_dir / "metal_prices.png")
    correlation_heatmap(df, output_dir / "correlation_heatmap.png")
    
    print(f"\nWykresy zostały zapisane w katalogu: {output_dir}")

if __name__ == "__main__":
    main()
