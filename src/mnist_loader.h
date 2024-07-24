// mnist_loader.h

#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <fstream>
#include <vector>
#include <string>

int32_t read_int(std::ifstream &file);
std::vector<std::vector<uint8_t>> load_mnist_images(const std::string &file_path);
std::vector<uint8_t> load_mnist_labels(const std::string &file_path);

#endif // MNIST_LOADER_H
