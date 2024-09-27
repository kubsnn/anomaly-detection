# Detekcja anomalii z wykorzystaniem Autodekoderów i sieci U-Net

-- OPIS TODO --

## Wymagania

Aby uruchomić ten projekt, potrzebujesz:
- Python 3.10 lub nowszy
- `make`
- Zainstalowane `pip` (Python Package Installer)

## Instalacja i Użycie

### Tworzenie Środowiska i Instalacja Zależności

Aby utworzyć środowisko wirtualne i zainstalować wszystkie zależności, wykonaj poniższe polecenie w terminalu:

```bash
make env
```
To polecenie tworzy środowisko wirtualne w katalogu venv.
Instaluje wszystkie zależności wymienione w pliku requirements.txt oraz instaluje najbardziej pasującą wersję PyTorch.

### Uruchamianie Aplikacji

Aby uruchomić aplikację, użyj następującego polecenia:

```bash
make run
```
To polecenie uruchamia główny skrypt aplikacji main.py w kontekście środowiska wirtualnego.

### Czyszczenie Środowiska

Aby usunąć środowisko wirtualne i inne pliki tymczasowe, użyj polecenia:

```bash
make clean
```
To polecenie usuwa katalog venv i jego zawartość.