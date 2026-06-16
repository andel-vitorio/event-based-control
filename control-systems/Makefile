# Configurações do compilador
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude -Iinc -Ilib

# Bibliotecas (Código fonte em src/)
LIB_SRCS := $(wildcard src/*.cpp)
LIB_OBJS := $(LIB_SRCS:.cpp=.o)

# Executáveis (Arquivos .cpp na raiz, excluindo os de src/)
MAIN_SRCS := $(filter-out src/%, $(wildcard *.cpp))
TARGETS := $(MAIN_SRCS:.cpp=.exe)

# Regra padrão
all: $(TARGETS)

# Regra para gerar os executáveis
# $(TARGETS) depende dos objetos da lib e do seu próprio objeto main
%.exe: %.o $(LIB_OBJS)
	$(CXX) $^ -o $@

# Regra para compilar o main.cpp (ou qualquer outro na raiz)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Regra para compilar os arquivos de src/
src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Limpeza
clean:
	del /Q *.o src\*.o *.exe