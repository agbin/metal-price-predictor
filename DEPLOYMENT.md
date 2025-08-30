# Instrukcja wdrożenia aplikacji na Digital Ocean

## Przygotowanie

Aplikacja Metal Predictor jest przygotowana do wdrożenia w kontenerze Docker, co ułatwia instalację i uruchomienie na Digital Ocean.

## Wymagania
- Konto na Digital Ocean
- Zainstalowany Docker i Docker Compose (na maszynie lokalnej)
- Git

## Krok 1: Utworzenie Droplet na Digital Ocean

1. Zaloguj się do [Digital Ocean](https://www.digitalocean.com/)
2. Kliknij "Create" i wybierz "Droplets"
3. Wybierz obraz "Marketplace" i wyszukaj "Docker"
4. Wybierz rozmiar: Rekomendowany Basic z 2 GB RAM / 1 CPU
5. Wybierz preferowany region (najlepiej najbliższy geograficznie)
6. Dodaj swój klucz SSH lub ustaw hasło
7. Nadaj nazwę swojemu Droplet, np. "metal-predictor"
8. Kliknij "Create Droplet"

## Krok 2: Połączenie z serwerem

```bash
ssh root@TWÓJ_IP_ADRES
```

## Krok 3: Klonowanie repozytorium

```bash
git clone https://github.com/TWOJE_REPOZYTORIUM/metal_predictor.git
cd metal_predictor
```

Alternatywnie, możesz przesłać pliki za pomocą SCP:
```bash
scp -r /ścieżka/do/metal_predictor root@TWÓJ_IP_ADRES:/root/
```

## Krok 4: Uruchomienie aplikacji

```bash
cd metal_predictor
docker-compose up -d
```

## Krok 5: Dostęp do aplikacji

Aplikacja będzie dostępna pod adresem:
```
http://TWÓJ_IP_ADRES:8501
```

## Krok 6: Monitorowanie logów

```bash
docker-compose logs -f
```

## Krok 7: Zatrzymanie aplikacji (jeśli potrzebne)

```bash
docker-compose down
```

## Rozwiązywanie problemów

### Problem z pamięcią
Jeśli Droplet ma za mało pamięci, możesz zwiększyć jego rozmiar w panelu Digital Ocean lub dodać plik wymiany:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Problem z dostępem
Upewnij się, że port 8501 jest otwarty w firewallu:

```bash
sudo ufw allow 8501/tcp
```

### Problem z danymi
Jeśli aplikacja nie może pobrać danych, sprawdź logi:

```bash
docker-compose logs -f
```
