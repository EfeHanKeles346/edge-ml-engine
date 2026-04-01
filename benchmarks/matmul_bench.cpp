#include "edge_ml/tensor.h"
#include <chrono>
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

double benchmark(const std::string& name, std::function<void()> fn, int runs = 5) {
    // Warmup
    fn();

    double total = 0;
    for (int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        fn();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        total += ms;
    }
    double avg = total / runs;
    std::cout << name << ": " << avg << " ms (avg of " << runs << " runs)" << std::endl;
    return avg;
}

int main() {
    std::cout << "=== Matmul Benchmark ===" << std::endl << std::endl;

    int sizes[] = {128, 256, 512, 1000};

    for (int n : sizes) {
        std::cout << "--- " << n << "x" << n << " ---" << std::endl;
        Tensor a = random_tensor({n, n});
        Tensor b = random_tensor({n, n});

        double naive_ms = benchmark("  Naive ", [&]() { a.matmul(b); });
        double tiled_ms = benchmark("  Tiled ", [&]() { a.matmul_tiled(b); });

        double speedup = naive_ms / tiled_ms;
        std::cout << "  Speedup: " << speedup << "x" << std::endl << std::endl;
    }

    return 0;
}
