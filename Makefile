# Makefile

# Nazwa katalogu środowiska wirtualnego
VENV_DIR = venv

# Plik z zależnościami
REQUIREMENTS = requirements.txt

# Detekcja systemu operacyjnego
ifeq ($(OS),Windows_NT)
    PLATFORM = windows
    PYTHON = $(VENV_DIR)\Scripts\python.exe
    PIP = $(VENV_DIR)\Scripts\pip.exe
    RM = rmdir /S /Q $(VENV_DIR)
else
    PLATFORM = unix
    PYTHON = $(VENV_DIR)/bin/python
    PIP = $(VENV_DIR)/bin/pip
    RM = rm -rf $(VENV_DIR)
endif

# Deklaracja celów, które nie są plikami
.PHONY: env clean run

# Cel: env - tworzy środowisko wirtualne i instaluje zależności
env: 
	python setup.py create

	$(PYTHON) setup.py install

	@echo "Virtual environment has been created and dependencies have been installed."

# Cel: clean - usuwa środowisko wirtualne
clean:
	@echo "Cleaning up..."
	$(RM)
	@echo "Done."

# Cel: run - uruchamia aplikację w środowisku wirtualnym bez aktywacji
run:
	@echo "Starting the application..."
	@$(PYTHON) src/main.py || { echo "Błąd: Nie udało się uruchomić aplikacji."; exit 1; }
