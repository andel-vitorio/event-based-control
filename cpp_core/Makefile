# --- Configurações ---
CXX      := g++
CXXFLAGS := -std=c++17 -O3 -Wall -fopenmp -Iinclude
LDFLAGS  := -fopenmp

# Diretórios
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := ../bin

# --- Lógica de Seleção de Arquivos ---

# 1. Fontes da Biblioteca (Tudo dentro de src/)
# Estes arquivos serão compilados mas NÃO geram executáveis sozinhos
LIB_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
# Transforma src/Numeric.cpp -> obj/Numeric.o
LIB_OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(LIB_SRCS))

# 2. Fontes Principais (Tudo na raiz do cpp_core/)
# Cada um destes arquivos virará um .exe separado
MAIN_SRCS := $(wildcard *.cpp)
# Transforma simulator.cpp -> ../bin/simulator.exe
EXECUTABLES := $(patsubst %.cpp, $(BIN_DIR)/%.exe, $(MAIN_SRCS))

# --- Regras de Build ---

.PHONY: all clean create_dirs info

# Regra padrão: gera todos os executáveis encontrados na raiz
all: create_dirs $(EXECUTABLES)

# Regra de Linkagem (Cria o .exe)
# Pega o objeto do main específico + TODOS os objetos da biblioteca
$(BIN_DIR)/%.exe: $(OBJ_DIR)/%.o $(LIB_OBJS)
	@echo [LINK] Gerando $@...
	$(CXX) $^ -o $@ $(LDFLAGS)

# Compilação dos MAINS (Raiz -> obj/)
$(OBJ_DIR)/%.o: %.cpp
	@echo [COMPILE MAIN] $<...
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compilação das LIBS (src/ -> obj/)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo [COMPILE LIB] $<...
	$(CXX) $(CXXFLAGS) -c $< -o $@

# --- Utilitários ---

create_dirs:
	@if not exist "$(OBJ_DIR)" mkdir "$(OBJ_DIR)"
	@if not exist "$(BIN_DIR)" mkdir "$(BIN_DIR)"

clean:
	@if exist "$(OBJ_DIR)" rmdir /s /q "$(OBJ_DIR)"
	@if exist "$(BIN_DIR)\*.exe" del /q "$(BIN_DIR)\*.exe"
	@echo [CLEAN] Limpeza concluida.

info:
	@echo Libs: $(LIB_SRCS)
	@echo Mains: $(MAIN_SRCS)
	@echo Exes: $(EXECUTABLES)