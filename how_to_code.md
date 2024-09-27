# Zasady Stylu Kodowania dla Pythona

Zasady stylu kodowania dla Pythona, które pomogą napisać czytelny, spójny i zgodny z najlepszymi praktykami kod. Te zasady bazują głównie na [PEP 8](https://pep8.org/), oficjalnym przewodniku stylu dla języka Python, oraz na dodatkowych najlepszych praktykach programistycznych.

## 1. Nazewnictwo (Naming Conventions)

### Nazwy zmiennych i funkcji
- **Charakterystyka**: Używaj małych liter, oddzielając słowa podkreśleniami.
- **Styl**: `snake_case`
- **Przykład**: `data_loader`, `calculate_mean()`

### Nazwy klas
- **Charakterystyka**: Używaj wielkich liter na początku każdego słowa (PascalCase).
- **Styl**: `PascalCase`
- **Przykład**: `DataProcessor`, `AutoencoderModel`

### Nazwy stałych
- **Charakterystyka**: Używaj wielkich liter, oddzielając słowa podkreśleniami.
- **Styl**: `UPPER_SNAKE_CASE`
- **Przykład**: `MAX_ITERATIONS`, `LEARNING_RATE`

### Nazwy modułów i pakietów
- **Charakterystyka**: Używaj małych liter, unikaj podkreśleń.
- **Styl**: `snake_case` lub `lowercase`
- **Przykład**: `data_processing`, `models`

## 2. Wcięcia (Indentation)
- **Charakterystyka**: Używaj 4 spacji na poziom wcięcia.
- **Styl**: Stałe wcięcia.
- **Przykład**:
  ```python
  def example_function():
      if condition:
          do_something()

## 3. Długość linii (Line Length)
- **Charakterystyka**: Ogranicz długość linii do 79 znaków.
- **Styl**: Jeśli linia jest zbyt długa, użyj podziału na mniejsze części, np. z użyciem nawiasów.
- **Przykład**:
  ```python
  result = some_function_with_a_long_name(
      parameter_one, parameter_two, parameter_three
  )

## 4. Białe znaki (Whitespace)

### Przestrzeń wokół operatorów
- **Charakterystyka**: Używaj pojedynczych spacji wokół operatorów przypisania i arytmetycznych.
- **Styl**: `a = b + c`

### Przestrzeń wewnątrz nawiasów
- **Charakterystyka**: Unikaj dodatkowych spacji bezpośrednio wewnątrz nawiasów.
- **Styl**: `func(a, b)` zamiast `func( a, b )`

### Puste linie
- **Charakterystyka**: Używaj pustych linii do oddzielenia funkcji i klas oraz większych bloków kodu wewnątrz funkcji.
- **Styl**: Dwie puste linie przed definicją funkcji lub klasy na najwyższym poziomie, jedna pusta linia wewnątrz funkcji.