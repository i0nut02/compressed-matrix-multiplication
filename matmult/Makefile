CXX = g++-15
NVCC = nvcc
CXXFLAGS = -std=c++17 -Iinclude -Wall -Wextra -O2
NVCCFLAGS = -std=c++17 -Iinclude -O2

SRC_DIR = src
BIN_DIR = bin
BUILD_DIR = build

CPP_SOURCES = $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*/*.cpp)
CUDA_SOURCES = $(wildcard $(SRC_DIR)/*.cu $(SRC_DIR)/*/*.cu)
CPP_OBJECTS = $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)
OBJECTS = $(CPP_OBJECTS) $(CUDA_OBJECTS)

TARGET = $(BIN_DIR)/main

$(TARGET): $(OBJECTS) | $(BIN_DIR)
    $(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
    @mkdir -p $(dir $@)
    $(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
    @mkdir -p $(dir $@)
    $(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN_DIR) $(BUILD_DIR):
    mkdir -p $@

clean:
    rm -rf $(BIN_DIR) $(BUILD_DIR)

.PHONY: clean

print-%:
    @echo $* = $($*)