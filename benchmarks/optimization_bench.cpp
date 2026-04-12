#include "edge_ml/tensor.h"
#include "edge_ml/operators/activations.h"
#include "edge_ml/optimizer/simd_ops.h"
#include "edge_ml/optimizer/quantization.h"
#include <chrono>
#include <functional>
#include <iostream>
#include <random>

using namespace edge_ml;

Tensor random_tensor(const std::vector<int>& shape) {
    Tensor t(shape);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < t.size(); i++) {
        t[i] = dist(gen);
    }
    return t;
}

double benchmark(const std::string& name, std::function<void()> fn, int runs = 10) {
    fn(); // warmup

    double total = 0;
    for (int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        fn();
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double avg = total / runs;
    std::cout << "  " << name << ": " << avg << " ms" << std::endl;
    return avg;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Edge ML Engine - Optimization Benchmark" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;

    // --- Element-wise Add ---
    {
        std::cout << "--- Element-wise Add (1M elements) ---" << std::endl;
        Tensor a = random_tensor({1000, 1000});
        Tensor b = random_tensor({1000, 1000});

        double naive = benchmark("Naive add", [&]() { a.add(b); });
        double simd = benchmark("SIMD  add", [&]() { opt::simd_add(a, b); });
        std::cout << "  Speedup: " << naive / simd << "x" << std::endl << std::endl;
    }

    // --- Element-wise Multiply ---
    {
        std::cout << "--- Element-wise Multiply (1M elements) ---" << std::endl;
        Tensor a = random_tensor({1000, 1000});
        Tensor b = random_tensor({1000, 1000});

        double naive = benchmark("Naive mul", [&]() { a.multiply(b); });
        double simd = benchmark("SIMD  mul", [&]() { opt::simd_multiply(a, b); });
        std::cout << "  Speedup: " << naive / simd << "x" << std::endl << std::endl;
    }

    // --- ReLU ---
    {
        std::cout << "--- ReLU (1M elements) ---" << std::endl;
        Tensor a = random_tensor({1000, 1000});

        double naive = benchmark("Naive relu", [&]() { ops::relu(a); });
        double simd = benchmark("SIMD  relu", [&]() { opt::simd_relu(a); });
        std::cout << "  Speedup: " << naive / simd << "x" << std::endl << std::endl;
    }

    // --- Matrix Multiplication ---
    {
        int sizes[] = {128, 256, 512};
        for (int n : sizes) {
            std::cout << "--- Matmul " << n << "x" << n << " ---" << std::endl;
            Tensor a = random_tensor({n, n});
            Tensor b = random_tensor({n, n});

            double naive = benchmark("Naive matmul", [&]() { a.matmul(b); });
            double tiled = benchmark("Tiled matmul", [&]() { a.matmul_tiled(b); });
            double simd = benchmark("SIMD  matmul", [&]() { opt::simd_matmul(a, b); });
            std::cout << "  Speedup (SIMD vs naive): " << naive / simd << "x" << std::endl;
            std::cout << "  Speedup (SIMD vs tiled): " << tiled / simd << "x" << std::endl << std::endl;
        }
    }

    // --- Quantization ---
    {
        std::cout << "--- Quantized Matmul 256x256 ---" << std::endl;
        Tensor a = random_tensor({256, 256});
        Tensor b = random_tensor({256, 256});

        double fp32 = benchmark("FP32 matmul", [&]() { a.matmul(b); });
        double int8 = benchmark("INT8 matmul", [&]() { opt::quantized_matmul(a, b); });

        // Accuracy comparison
        Tensor fp32_result = a.matmul(b);
        Tensor int8_result = opt::quantized_matmul(a, b);
        float max_error = 0;
        for (int i = 0; i < fp32_result.size(); i++) {
            float err = std::abs(fp32_result[i] - int8_result[i]);
            if (err > max_error) max_error = err;
        }
        std::cout << "  Max absolute error: " << max_error << std::endl;
        std::cout << std::endl;
    }

    // --- FP16 Simulation ---
    {
        std::cout << "--- FP16 Precision Loss ---" << std::endl;
        Tensor a = random_tensor({1000});
        Tensor fp16 = opt::simulate_fp16(a);
        float max_error = 0;
        for (int i = 0; i < a.size(); i++) {
            float err = std::abs(a[i] - fp16[i]);
            if (err > max_error) max_error = err;
        }
        std::cout << "  Max FP16 precision loss: " << max_error << std::endl;
        std::cout << std::endl;
    }

    // --- Memory Usage ---
    {
        std::cout << "--- Memory Comparison ---" << std::endl;
        int n = 1000000;
        Tensor t = random_tensor({n});
        auto qt = opt::quantize_int8(t);

        std::cout << "  FP32: " << n * 4 / 1024 << " KB" << std::endl;
        std::cout << "  INT8: " << n * 1 / 1024 << " KB" << std::endl;
        std::cout << "  Compression: " << 4.0 << "x" << std::endl;
    }

    std::cout << std::endl << "========================================" << std::endl;
    return 0;
}
