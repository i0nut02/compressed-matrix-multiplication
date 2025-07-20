CXX = g++-15
CXXFLAGS = -std=c++17 -Iinclude -Wall -Wextra -O2
SRC = $(wildcard src/*.cpp)
BIN_DIR = ./bin
TARGET = $(BIN_DIR)/format_tester

$(TARGET): $(SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

.PHONY: clean