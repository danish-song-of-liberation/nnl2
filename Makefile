CC = gcc
CFLAGS = -Wall -Wextra -Werror -O3 -fPIC -mavx -mavx2 -msse4.1 -msse4.2 

SRC = src/c/nnl2_core.c
OBJ = $(SRC:.c=.o)

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

OPENBLAS0330WOA64STATIC_DIRECTORY := $(MAKEFILE_DIR)backends/OpenBLAS-0.3.30-woa64-64-static/OpenBLAS

OPENBLAS0330WOA64STATIC_INCLUDE := $(OPENBLAS0330WOA64STATIC_DIRECTORY)/include/openblas64
OPENBLAS0330WOA64STATIC_LIB := $(OPENBLAS0330WOA64STATIC_DIRECTORY)/lib
OPENBLAS0330WOA64STATIC_SHARED := openblas

ifeq ($(OS),Windows_NT)
    TARGET = src/c/libnnl.dll
    LIBSUFFIX = .dll
    LDFLAGS = -shared
else
    TARGET = src/c/libnnl.so
    LIBSUFFIX = .so
    LDFLAGS = -shared 
endif

ifeq ($(openblas0330woa64static_available), 1)
	CFLAGS += -I$(OPENBLAS0330WOA64STATIC_INCLUDE)
	CFLAGS += -DOPENBLAS_AVAILABLE
    LDFLAGS += -L$(OPENBLAS0330WOA64STATIC_LIB) -l$(OPENBLAS0330WOA64STATIC_SHARED)
endif

ifeq ($(avx256_available), 1)
	CFLAGS += -DNNL2_AVX256_AVAILABLE
endif

all: $(TARGET)
	@echo OS=$(OS)
	@echo Building $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

$(OBJ): $(SRC)
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	git clean -fdX -- $(MAKEFILE_DIR)src/c/nnl2_core.o

.PHONY: all clean
