# 🚀 Metal Price Predictor - Quick Start

## ⚡ Szybkie uruchomienie (4 kroki)

### 1. Sklonuj repozytorium
```bash
git clone https://github.com/agbin/metal-price-predictor.git
cd metal-price-predictor
```

### 2. Utwórz środowisko Python (opcjonalnie, ale zalecane)
```bash
python -m venv metal_env
source metal_env/bin/activate  # Linux/Mac
# lub
metal_env\Scripts\activate     # Windows
```

### 3. Zainstaluj biblioteki
```bash
pip install -r requirements.txt
```

### 4. Uruchom aplikację
```bash
streamlit run app.py
```

**Aplikacja otworzy się w przeglądarce na: http://localhost:8501** 🎉

---

## 📊 Jak używać aplikacji

1. **Trenowanie modelu**: Wybierz metal → kliknij "Trenuj model" (trwa ~2 min)
2. **Prognozowanie**: Przejdź do zakładki "Prediction" → "Prognozuj cenę"
3. **Wyniki**: Zobacz wykresy i metryki w zakładce "Accuracy"

---

## 🔧 Rozwiązywanie problemów

### Problem: `ModuleNotFoundError`
**Rozwiązanie:** Zainstaluj wszystkie biblioteki:
```bash
pip install -r requirements.txt
```

### Problem: Konflikty wersji
**Rozwiązanie:** Użyj środowiska wirtualnego (krok 2 powyżej)

### Problem: Aplikacja się nie uruchamia
**Rozwiązanie:** Sprawdź czy jesteś w katalogu projektu:
```bash
ls -la  # Powinieneś zobaczyć app.py
```

---

## 💡 Co robi aplikacja?

**Metal Price Predictor** to system ML do prognozowania cen metali szlachetnych:
- 📈 **Dane**: Yahoo Finance, FRED, World Bank (2000-2025)
- 🤖 **ML**: PyCaret, autoregresja, lag features
- 📊 **UI**: Streamlit z wykresami Plotly
- 🎯 **Metale**: Złoto, Srebro, Platyna, Pallad, Miedź

---

## 📞 Wsparcie

W razie problemów sprawdź logi w terminalu lub otwórz issue na GitHub.

**Status:** ✅ Działająca wersja
