#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <fstream>
#include "mnist_loader.h"

struct VAEParams {
    float *encoder_weights;
    float *decoder_weights;
    float *encoder_biases;
    float *decoder_biases;
    int input_dim;
    int hidden_dim;
    int latent_dim;
};

__device__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void forwardPass(VAEParams params, float* input, float* output, float* mean, float* log_var, float* z) {
    int idx = threadIdx.x;
    if (idx >= params.input_dim) return;

    // Encoder
    // First Layer, Hidden Representation from Input
    for (int i = 0; i < params.hidden_dim; i++) {
        float sum = params.encoder_biases[i];
        for (int j = 0; j < params.input_dim; j++) {
            sum += input[j] * params.encoder_weights[i * params.input_dim + j];
        }
        mean[i] = relu(sum);
    }

    // Second Layer, Mean and Log Variance
    for (int i = 0; i < params.latent_dim; i++) {
        float sum_mean = params.encoder_biases[params.hidden_dim + i];
        float sum_log_var = params.encoder_biases[params.hidden_dim + params.latent_dim + i];
        for (int j = 0; j < params.hidden_dim; j++) {
            sum_mean += mean[j] * params.encoder_weights[(params.hidden_dim + i) * params.input_dim + j];
            sum_log_var += mean[j] * params.encoder_weights[(params.hidden_dim + params.latent_dim + i) * params.input_dim + j];
        }
        mean[i] = sum_mean;
        log_var[i] = sum_log_var;
    }

    // Reparameterization trick
    curandState state;
    curand_init(0, idx, 0, &state);
    for (int i = 0; i < params.latent_dim; i++) {
        float epsilon = curand_normal(&state);
        z[i] = mean[i] + expf(0.5f * log_var[i]) * epsilon;
    }

    // Decoder
    // First Layer, Hidden Representation from Latent Variables
    for (int i = 0; i < params.hidden_dim; i++) {
        float sum = params.decoder_biases[i];
        for (int j = 0; j < params.latent_dim; j++) {
            sum += z[j] * params.decoder_weights[i * params.latent_dim + j];
        }
        output[i] = relu(sum);
    }

    // Second Layer, Reconstruction
    for (int i = 0; i < params.input_dim; i++) {
        float sum = params.decoder_biases[params.hidden_dim + i];
        for (int j = 0; j < params.hidden_dim; j++) {
            sum += output[j] * params.decoder_weights[(params.hidden_dim + i) * params.latent_dim + j];
        }
        output[i] = sigmoid(sum);
    }
}

__global__ void backwardPass(VAEParams params, float* input, float* output, float* mean, float* log_var, float* z, float* gradients) {
    int idx = threadIdx.x;
    if (idx >= params.input_dim) return;

    // Calculate output layer gradients
    float output_grad = output[idx] - input[idx];

    // Gradients for decoder weights and biases
    for (int i = 0; i < params.hidden_dim; i++) {
        float decoder_grad = output_grad * (output[i] * (1 - output[i])); // derivative of sigmoid
        atomicAdd(&gradients[i * params.input_dim + idx], decoder_grad);
        atomicAdd(&params.decoder_biases[idx], decoder_grad);
    }

    // Backpropagate to latent space
    float z_grad = 0.0f;
    for (int i = 0; i < params.latent_dim; i++) {
        for (int j = 0; j < params.hidden_dim; j++) {
            float decoder_weight = params.decoder_weights[j * params.latent_dim + i];
            z_grad += decoder_weight * output_grad * (output[j] * (1 - output[j]));
        }
    }

    // Gradients for encoder weights and biases
    for (int i = 0; i < params.hidden_dim; i++) {
        float mean_grad = z_grad * expf(0.5f * log_var[i]) + mean[i];
        float log_var_grad = 0.5f * z_grad * expf(0.5f * log_var[i]) * (z[i] - mean[i]);

        atomicAdd(&gradients[i * params.input_dim + idx], mean_grad);
        atomicAdd(&gradients[(params.hidden_dim + i) * params.input_dim + idx], log_var_grad);

        atomicAdd(&params.encoder_biases[i], mean_grad);
        atomicAdd(&params.encoder_biases[params.hidden_dim + i], log_var_grad);
    }
}

__global__ void updateWeights(VAEParams params, float* gradients, float learning_rate) {
    int idx = threadIdx.x;
    if (idx >= params.input_dim) return;

    // Simple weight update
    for (int i = 0; i < params.hidden_dim; i++) {
        params.encoder_weights[i * params.input_dim + idx] -= learning_rate * gradients[i * params.input_dim + idx];
        params.decoder_weights[i * params.latent_dim + idx] -= learning_rate * gradients[i * params.latent_dim + idx];
    }

    for (int i = 0; i < params.latent_dim; i++) {
        params.encoder_biases[i] -= learning_rate * gradients[params.hidden_dim * params.input_dim + i];
        params.decoder_biases[i] -= learning_rate * gradients[params.hidden_dim * params.latent_dim + i];
    }
}

__global__ void calculateLoss(float* input, float* output, float* mean, float* log_var, float* loss, int size) {
    int idx = threadIdx.x;
    if (idx >= size) return;

    float diff = input[idx] - output[idx];
    atomicAdd(loss, diff * diff); // Sum of squared differences

    // KL divergence loss
    if (idx < size) {
        float kl_loss = 0.5f * (expf(log_var[idx]) + mean[idx] * mean[idx] - 1.0f - log_var[idx]);
        atomicAdd(loss, kl_loss);
    }
}

void initializeWeights(float* weights, int size) {
    for (int i = 0; i < size; i++) {
        weights[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
}

void checkCUDAError(const char *message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

void saveModel(const VAEParams& params, int hidden_dim, int input_dim, int latent_dim) {
    std::ofstream file("model.pth", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open model file for saving." << std::endl;
        return;
    }

    int encoder_weights_size = hidden_dim * input_dim;
    int decoder_weights_size = hidden_dim * latent_dim;
    int encoder_biases_size = hidden_dim + 2 * latent_dim;
    int decoder_biases_size = hidden_dim + input_dim;

    float* h_encoder_weights = new float[encoder_weights_size];
    float* h_decoder_weights = new float[decoder_weights_size];
    float* h_encoder_biases = new float[encoder_biases_size];
    float* h_decoder_biases = new float[decoder_biases_size];

    cudaMemcpy(h_encoder_weights, params.encoder_weights, encoder_weights_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_decoder_weights, params.decoder_weights, decoder_weights_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_encoder_biases, params.encoder_biases, encoder_biases_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_decoder_biases, params.decoder_biases, decoder_biases_size * sizeof(float), cudaMemcpyDeviceToHost);

    file.write(reinterpret_cast<char*>(h_encoder_weights), encoder_weights_size * sizeof(float));
    file.write(reinterpret_cast<char*>(h_decoder_weights), decoder_weights_size * sizeof(float));
    file.write(reinterpret_cast<char*>(h_encoder_biases), encoder_biases_size * sizeof(float));
    file.write(reinterpret_cast<char*>(h_decoder_biases), decoder_biases_size * sizeof(float));

    delete[] h_encoder_weights;
    delete[] h_decoder_weights;
    delete[] h_encoder_biases;
    delete[] h_decoder_biases;

    file.close();
}

bool loadModel(VAEParams& params, int hidden_dim, int input_dim, int latent_dim) {
    std::ifstream file("model.pth", std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    int encoder_weights_size = hidden_dim * input_dim;
    int decoder_weights_size = hidden_dim * latent_dim;
    int encoder_biases_size = hidden_dim + 2 * latent_dim;
    int decoder_biases_size = hidden_dim + input_dim;

    float* h_encoder_weights = new float[encoder_weights_size];
    float* h_decoder_weights = new float[decoder_weights_size];
    float* h_encoder_biases = new float[encoder_biases_size];
    float* h_decoder_biases = new float[decoder_biases_size];

    file.read(reinterpret_cast<char*>(h_encoder_weights), encoder_weights_size * sizeof(float));
    file.read(reinterpret_cast<char*>(h_decoder_weights), decoder_weights_size * sizeof(float));
    file.read(reinterpret_cast<char*>(h_encoder_biases), encoder_biases_size * sizeof(float));
    file.read(reinterpret_cast<char*>(h_decoder_biases), decoder_biases_size * sizeof(float));

    cudaMemcpy(params.encoder_weights, h_encoder_weights, encoder_weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(params.decoder_weights, h_decoder_weights, decoder_weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(params.encoder_biases, h_encoder_biases, encoder_biases_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(params.decoder_biases, h_decoder_biases, decoder_biases_size * sizeof(float), cudaMemcpyHostToDevice);

    delete[] h_encoder_weights;
    delete[] h_decoder_weights;
    delete[] h_encoder_biases;
    delete[] h_decoder_biases;

    file.close();
    return true;
}

void loadData(std::vector<std::vector<uint8_t>>& images, float* data, int data_size, int input_dim) {
    for (int i = 0; i < data_size; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            data[i * input_dim + j] = static_cast<float>(images[i][j]) / 255.0f;
        }
    }
}

int main() {
    VAEParams params;
    params.input_dim = 784; // Example input dimension for MNIST dataset
    params.hidden_dim = 400;
    params.latent_dim = 20;
    int data_size = 60000; // Number of training samples in MNIST
    float learning_rate = 0.001;
    int epochs = 10; // Training epochs

    // Load MNIST data
    std::vector<std::vector<uint8_t>> train_images = load_mnist_images("../data/MNIST/raw/train-images-idx3-ubyte");
    std::vector<uint8_t> train_labels = load_mnist_labels("../data/MNIST/raw/train-labels-idx1-ubyte");
    
    // Allocate host memory for training data
    float *h_data = (float*)malloc(data_size * params.input_dim * sizeof(float));
    loadData(train_images, h_data, data_size, params.input_dim);

    // Allocate device memory
    float *d_input, *d_output, *d_mean, *d_log_var, *d_z, *d_gradients, *d_loss;
    float h_loss = 0.0f;

    cudaMalloc(&d_input, params.input_dim * sizeof(float));
    checkCUDAError("CUDA malloc failed for d_input");
    cudaMalloc(&d_output, params.input_dim * sizeof(float));
    checkCUDAError("CUDA malloc failed for d_output");
    cudaMalloc(&d_mean, params.latent_dim * sizeof(float));
    checkCUDAError("CUDA malloc failed for d_mean");
    cudaMalloc(&d_log_var, params.latent_dim * sizeof(float));
    checkCUDAError("CUDA malloc failed for d_log_var");
    cudaMalloc(&d_z, params.latent_dim * sizeof(float));
    checkCUDAError("CUDA malloc failed for d_z");
    cudaMalloc(&d_gradients, params.input_dim * sizeof(float));
    checkCUDAError("CUDA malloc failed for d_gradients");
    cudaMalloc(&d_loss, sizeof(float));
    checkCUDAError("CUDA malloc failed for d_loss");

    cudaMalloc(&params.encoder_weights, params.hidden_dim * params.input_dim * sizeof(float));
    checkCUDAError("CUDA malloc failed for params.encoder_weights");
    cudaMalloc(&params.decoder_weights, params.hidden_dim * params.latent_dim * sizeof(float));
    checkCUDAError("CUDA malloc failed for params.decoder_weights");
    cudaMalloc(&params.encoder_biases, (params.hidden_dim + 2 * params.latent_dim) * sizeof(float));
    checkCUDAError("CUDA malloc failed for params.encoder_biases");
    cudaMalloc(&params.decoder_biases, (params.hidden_dim + params.input_dim) * sizeof(float));
    checkCUDAError("CUDA malloc failed for params.decoder_biases");

    if (!loadModel(params, params.hidden_dim, params.input_dim, params.latent_dim)) {
        // Initialize weights
        float *h_encoder_weights = (float*)malloc(params.hidden_dim * params.input_dim * sizeof(float));
        float *h_decoder_weights = (float*)malloc(params.hidden_dim * params.latent_dim * sizeof(float));
        float *h_encoder_biases = (float*)malloc((params.hidden_dim + 2 * params.latent_dim) * sizeof(float));
        float *h_decoder_biases = (float*)malloc((params.hidden_dim + params.input_dim) * sizeof(float));

        initializeWeights(h_encoder_weights, params.hidden_dim * params.input_dim);
        initializeWeights(h_decoder_weights, params.hidden_dim * params.latent_dim);
        initializeWeights(h_encoder_biases, params.hidden_dim + 2 * params.latent_dim);
        initializeWeights(h_decoder_biases, params.hidden_dim + params.input_dim);

        cudaMemcpy(params.encoder_weights, h_encoder_weights, params.hidden_dim * params.input_dim * sizeof(float), cudaMemcpyHostToDevice);
        checkCUDAError("CUDA memcpy failed for params.encoder_weights");
        cudaMemcpy(params.decoder_weights, h_decoder_weights, params.hidden_dim * params.latent_dim * sizeof(float), cudaMemcpyHostToDevice);
        checkCUDAError("CUDA memcpy failed for params.decoder_weights");
        cudaMemcpy(params.encoder_biases, h_encoder_biases, (params.hidden_dim + 2 * params.latent_dim) * sizeof(float), cudaMemcpyHostToDevice);
        checkCUDAError("CUDA memcpy failed for params.encoder_biases");
        cudaMemcpy(params.decoder_biases, h_decoder_biases, (params.hidden_dim + params.input_dim) * sizeof(float), cudaMemcpyHostToDevice);
        checkCUDAError("CUDA memcpy failed for params.decoder_biases");

        free(h_encoder_weights);
        free(h_decoder_weights);
        free(h_encoder_biases);
        free(h_decoder_biases);
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        h_loss = 0.0f;

        for (int i = 0; i < data_size; i++) {
            cudaMemcpy(d_input, &h_data[i * params.input_dim], params.input_dim * sizeof(float), cudaMemcpyHostToDevice);
            checkCUDAError("CUDA memcpy failed for d_input");

            // Reset loss
            cudaMemset(d_loss, 0, sizeof(float));
            checkCUDAError("CUDA memset failed for d_loss");

            // Launch the forward and backward pass kernels
            forwardPass<<<1, params.input_dim>>>(params, d_input, d_output, d_mean, d_log_var, d_z);
            checkCUDAError("Forward Pass Kernel Execution");

            backwardPass<<<1, params.input_dim>>>(params, d_input, d_output, d_mean, d_log_var, d_z, d_gradients);
            checkCUDAError("Backward Pass Kernel Execution");

            updateWeights<<<1, params.input_dim>>>(params, d_gradients, learning_rate);
            checkCUDAError("Update Weights Kernel Execution");

            // Calculate loss
            calculateLoss<<<1, params.input_dim>>>(d_input, d_output, d_mean, d_log_var, d_loss, params.input_dim);
            checkCUDAError("Calculate Loss Kernel Execution");

            // Copy loss from device to host
            cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
            checkCUDAError("CUDA memcpy failed for d_loss");

            std::cout << "Epoch " << epoch << ", Sample " << i << ", Loss: " << h_loss << std::endl;
        }
    }

    saveModel(params, params.hidden_dim, params.input_dim, params.latent_dim);

    // Testing loop
    for (int i = 0; i < data_size; i++) {
        cudaMemcpy(d_input, &h_data[i * params.input_dim], params.input_dim * sizeof(float), cudaMemcpyHostToDevice);
        checkCUDAError("CUDA memcpy failed for d_input");

        // Launch the forward pass kernel
        forwardPass<<<1, params.input_dim>>>(params, d_input, d_output, d_mean, d_log_var, d_z);
        checkCUDAError("Forward Pass Kernel Execution");

        // Calculate loss
        cudaMemset(d_loss, 0, sizeof(float));
        checkCUDAError("CUDA memset failed for d_loss");

        calculateLoss<<<1, params.input_dim>>>(d_input, d_output, d_mean, d_log_var, d_loss, params.input_dim);
        checkCUDAError("Calculate Loss Kernel Execution");

        // Copy loss from device to host
        cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        checkCUDAError("CUDA memcpy failed for d_loss");

        std::cout << "Test Sample " << i << ", Loss: " << h_loss << std::endl;
    }

    // Free allocated memory
    free(h_data);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mean);
    cudaFree(d_log_var);
    cudaFree(d_z);
    cudaFree(d_gradients);
    cudaFree(d_loss);
    cudaFree(params.encoder_weights);
    cudaFree(params.decoder_weights);
    cudaFree(params.encoder_biases);
    cudaFree(params.decoder_biases);

    return 0;
}