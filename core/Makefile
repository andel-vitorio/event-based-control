# =====================================================
# Configuration
# =====================================================

CXX := g++

CXXFLAGS := -std=c++23
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -fPIC
CXXFLAGS += -Iinclude
CXXFLAGS += -Iexternal
CXXFLAGS += -Imodules/control-systems/include

OBJ_DIR := obj
DEP_DIR := d
BIN_DIR := bin
DLL_DIR := dll
INT_DIR := src/interfaces

APP ?= Main

ENTRY_SRC := app/$(APP).cpp
TARGET_MAIN := $(BIN_DIR)/$(APP).exe
TARGET_TEST := $(BIN_DIR)/TestRunner.exe

ifdef app
  APP := $(app)
endif

# =====================================================
# Source Discovery
# =====================================================

rwildcard = $(wildcard $1$2) $(foreach d,$(wildcard $1*),$(if $(wildcard $d/),$(call rwildcard,$d/,$2)))

# Varre automaticamente todas as subpastas a partir de src/ e modules/control-systems/src/
SRC_FILES := $(call rwildcard,src/,*.cpp) \
             $(call rwildcard,modules/control-systems/src/,*.cpp)

INTERFACE_SRCS := $(wildcard $(INT_DIR)/*.cpp)

TEST_SRCS := $(wildcard tests/*.cpp)

# Código reutilizável da biblioteca
CORE_SRCS := $(filter-out $(INTERFACE_SRCS),$(SRC_FILES))

# =====================================================
# Objects
# =====================================================

CORE_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(CORE_SRCS))

MAIN_OBJ := $(OBJ_DIR)/$(ENTRY_SRC:.cpp=.o)

TEST_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(TEST_SRCS))

INT_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(INTERFACE_SRCS))

OBJS_MAIN := $(MAIN_OBJ) $(CORE_OBJS)

OBJS_TEST := $(TEST_OBJS) $(CORE_OBJS)

DLL_TARGETS := \
	$(patsubst $(INT_DIR)/%.cpp,$(DLL_DIR)/%.dll,$(INTERFACE_SRCS))

DEPS := $(patsubst %.cpp,$(DEP_DIR)/%.d, $(ENTRY_SRC) $(CORE_SRCS) $(INTERFACE_SRCS) $(TEST_SRCS))

# =====================================================
# Phony Targets
# =====================================================

.PHONY: all tests dll clean clean-all clean-dlls

all: $(BIN_DIR) $(TARGET_MAIN)

tests: $(BIN_DIR) $(TARGET_TEST)
	@echo Running tests...
	@$(TARGET_TEST) $(if $(TEST_FILTER),-ts="$(TEST_FILTER)",)

dll: $(DLL_DIR) $(DLL_TARGETS)

# =====================================================
# Directories
# =====================================================

$(BIN_DIR):
	@if not exist "$@" mkdir "$@"

$(DLL_DIR):
	@if not exist "$@" mkdir "$@"

# =====================================================
# Executables
# =====================================================

$(TARGET_MAIN): $(OBJS_MAIN)
	@echo Linking $@
	$(CXX) $^ -o $@

$(TARGET_TEST): $(OBJS_TEST)
	@echo Linking $@
	$(CXX) $^ -o $@

# =====================================================
# DLL Generation
# =====================================================
define create_dll_rule
$(DLL_DIR)/$(notdir $(basename $(1))).dll: \
$(OBJ_DIR)/$(patsubst %.cpp,%.o,$(1)) \
$$(CORE_OBJS)

	@mkdir -p dll
	
	@echo Linking $$@
	$$(CXX) -shared $$^ \
		-o $$@ \
		-static-libgcc \
		-static-libstdc++ \
		-static
endef

$(foreach src,$(INTERFACE_SRCS),\
	$(eval $(call create_dll_rule,$(src))))

# =====================================================
# Compilation
# =====================================================

$(OBJ_DIR)/%.o: %.cpp
	@if not exist "$(dir $@)" mkdir "$(dir $@)"
	@if not exist "$(dir $(DEP_DIR)/$*)" mkdir "$(dir $(DEP_DIR)/$*)"
	@echo Compiling $<
	$(CXX) $(CXXFLAGS) \
		-MMD -MP \
		-MF $(DEP_DIR)/$*.d \
		-c $< \
		-o $@

# =====================================================
# Dependencies
# =====================================================

-include $(DEPS)

# =====================================================
# Maintenance
# =====================================================

clean:
	@if exist $(OBJ_DIR) rmdir /s /q $(OBJ_DIR)
	@if exist $(BIN_DIR) del /Q $(BIN_DIR)\*.exe

clean-dlls:
	@if exist $(DLL_DIR) rmdir /s /q $(DLL_DIR)

clean-all: clean clean-dlls
	@if exist $(DEP_DIR) rmdir /s /q $(DEP_DIR)