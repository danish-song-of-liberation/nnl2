CC = gcc
CFLAGS = -Wall -Wextra -Werror -O3 -fPIC

SRC = src/c/nnl2_core.c
OBJ = $(SRC:.c=.o)

OPENBLAS0330WOA64STATIC_DIRECTORY := backends/OpenBLAS-0.3.30-woa64-64-static/OpenBLAS

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
	CFLAGS += -L$(OPENBLAS0330WOA64STATIC_LIB)
	CFLAGS += -l$(OPENBLAS0330WOA64STATIC_SHARED)
	CFLAGS += -DOPENBLAS0330WOA64STATIC_AVAILABLE
endif

all: $(TARGET)
	@echo OS=$(OS)
	@echo Building $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

$(OBJ): $(SRC)
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJ) $(TARGET) *.o

.PHONY: all clean
