CC = gcc
CFLAGS = -Wall -Wextra -Werror -O3 -fPIC

SRC = src/c/nnl2_core.c
OBJ = $(SRC:.c=.o)

ifeq ($(OS),Windows_NT)
    TARGET = src/c/libnnl.dll
    LIBSUFFIX = .dll
    LDFLAGS = -shared
else
    TARGET = src/c/libnnl.so
    LIBSUFFIX = .so
    LDFLAGS = -shared
endif

all: $(TARGET)
	@echo OS=$(OS)
	@echo Building $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

$(OBJ): $(SRC)
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
