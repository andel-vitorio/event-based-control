# =====================================================
# Configuration
# =====================================================

CXX      := g++
CXXFLAGS := -std=c++26 -Wall -Wextra -Iinclude -Iinc -Ilib
CXXFLAGS += -Imodules/control_core/include

OBJ_DIR := obj
DEP_DIR := d

TARGET := Main.exe

# =====================================================
# Automated Source Discovery
# =====================================================

# Procura arquivos .cpp na raiz, em src/, e em subdiretórios (até 3 níveis)
# Isso funciona nativamente no Make de forma idêntica em Windows e Linux
SRCS := $(wildcard *.cpp) \
        $(wildcard libs/control-core/src/*.cpp) \
        $(wildcard libs/control-core/src/**/*.cpp)

# Converte os caminhos de fontes para objetos mantendo a estrutura interna
# Exemplo: src/subpasta/Parser.cpp -> obj/src/subpasta/Parser.o
OBJS := $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# O mapeamento dos arquivos .d espelha exatamente a árvore de objetos
DEPS := $(patsubst %.cpp, $(DEP_DIR)/%.d, $(SRCS))

# =====================================================
# Git Subtree Configuration
# =====================================================

SUBTREE_REMOTE := git@github.com:andel-vitorio/control-systems.git
SUBTREE_BRANCH := control-core-ready
SUBTREE_DIR    := libs/control-core
SRC_DIR_SOURCE := modules/control-core

# =====================================================
# Build Rules
# =====================================================

.PHONY: all clean clean-all
.PHONY: all clean clean-all subtree-pull subtree-push subtree-split

all: $(TARGET)

# Regra de ligação
$(TARGET): $(OBJS)
	@echo Linking $@
	$(CXX) $(OBJS) -o $@

# Regra de compilação estrita
# Cria as árvores de subdiretórios necessárias dinamicamente antes de compilar
$(OBJ_DIR)/%.o: %.cpp
	@if not exist "$(dir $@)" mkdir "$(dir $@)"
	@if not exist "$(dir $(DEP_DIR)/$*)" mkdir "$(dir $(DEP_DIR)/$*)"
	@echo Compiling $<
	$(CXX) $(CXXFLAGS) -MMD -MP -MF $(DEP_DIR)/$*.d -c $< -o $@

# =====================================================
# Dependency Management
# =====================================================

# Inclui recursivamente todos os arquivos de dependência encontrados nas subpastas de d/
-include $(DEPS)

# =====================================================
# Maintenance
# =====================================================

clean:
	@if exist $(OBJ_DIR) rmdir /s /q $(OBJ_DIR)
	@if exist $(TARGET) del /Q $(TARGET)

clean-all: clean
	@if exist $(DEP_DIR) rmdir /s /q $(DEP_DIR)