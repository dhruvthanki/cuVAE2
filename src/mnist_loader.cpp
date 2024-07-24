// mnist_loader.cpp

#include "mnist_loader.h"

int32_t read_int(std::ifstream &file) {
    int32_t result = 0;
    file.read(reinterpret_cast<char *>(&result), 4);
    return __builtin_bswap32(result); // Convert from big-endian to little-endian
}

std::vector<std::vector<uint8_t>> load_mnist_images(const std::string &file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    int32_t magic_number = read_int(file);
    int32_t num_images = read_int(file);
    int32_t num_rows = read_int(file);
    int32_t num_cols = read_int(file);

    std::vector<std::vector<uint8_t>> images(num_images, std::vector<uint8_t>(num_rows * num_cols));
    for (int i = 0; i < num_images; ++i) {
        file.read(reinterpret_cast<char *>(images[i].data()), num_rows * num_cols);
    }

    file.close();
    return images;
}

std::vector<uint8_t> load_mnist_labels(const std::string &file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    int32_t magic_number = read_int(file);
    int32_t num_labels = read_int(file);

    std::vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char *>(labels.data()), num_labels);

    file.close();
    return labels;
}
