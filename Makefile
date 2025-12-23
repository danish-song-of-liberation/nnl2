CC = gcc
CFLAGS = -Wall -Wextra -Werror -O3 -fPIC -mavx -mavx2 -msse4.1 -msse4.2 -mfma -DNNL2_PTHREAD_AVAILABLE -fopenmp -lm

SRC = src/c/nnl2_core.c
OBJ = $(SRC:.c=.o)

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

OPENBLAS_DIR := backends/OpenBLAS-0.3.30-woa64-64-static/OpenBLAS
OPENBLAS_INCLUDE := $(OPENBLAS_DIR)/include/openblas64
OPENBLAS_LIB := $(OPENBLAS_DIR)/lib

ifeq ($(OS),Windows_NT)
    TARGET = src/c/libnnl.dll
    LDFLAGS = -shared
else
    TARGET = src/c/libnnl.so
    LDFLAGS = -shared 
endif

ifeq ($(openblas0330woa64static_available), 1)
	CFLAGS += -I$(OPENBLAS_INCLUDE)
	CFLAGS += -DOPENBLAS_AVAILABLE
	CFLAGS += -DNNL2_NUM_THREADS=$(kernel_count)
    LDFLAGS += -L$(OPENBLAS_LIB) -l:libopenblas_haswellp-r0.3.30.dev.a
endif

ifeq ($(avx256_available), 1)
	CFLAGS += -DNNL2_AVX256_AVAILABLE
endif

all: $(TARGET)
	@echo Building $(TARGET) for $(OS)

$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

$(OBJ): $(SRC)
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	git clean -fdX -- src/c/nnl2_core.o

.PHONY: all clean
