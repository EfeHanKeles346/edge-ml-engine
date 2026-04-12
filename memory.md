# Edge ML Engine — Proje Hafızası

> Bu dosya projenin başından bu yana yapılan her şeyi, kullanılan kavramları ve mimarisi belgeliyor.
> Birisi bu dosyayı okuyunca hem ML kavramlarına hakim olacak hem de projeyi tepeden tırnağa anlayacak.

---

## Projenin Amacı

Sıfırdan bir **C++ inference engine** yazmak. PyTorch yok, TensorFlow yok, ONNXRuntime yok — her şey elle yazıldı. Engine, ONNX formatındaki neural network modellerini yükleyip çalıştırabiliyor.

**Pipeline:**
```
ONNX Model (.onnx) → Protobuf Parser → Computation Graph (DAG) → Topological Sort → Executor → Output
```

**Nihai Hedef:** YOLOv11 object detection modelini bu engine üzerinde çalıştırmak.

---

## Temel Kavramlar

### Tensor Nedir?

Tensor, çok boyutlu bir sayı dizisidir. Neural network'lerin temel veri yapısıdır.

- **Skaler:** tek sayı → `5.0` (0 boyutlu)
- **Vektör:** 1D dizi → `[1, 2, 3]` (1 boyutlu)
- **Matris:** 2D tablo → `[[1,2],[3,4]]` (2 boyutlu)
- **Tensor:** 3D+ → Görüntü `[3, 224, 224]` = 3 kanal × 224 yükseklik × 224 genişlik

**Shape:** Tensor'ın boyutlarını tanımlar. `{2, 3, 4}` = 2×3×4 = 24 eleman.

**Stride:** Bellekte bir boyutta bir adım ilerlemek için kaç eleman atlanacağını söyler. Row-major layout'ta (C-style), son boyutun stride'ı 1'dir.
- Shape `{2, 3}` → Stride `{3, 1}` (satır atlamak için 3 eleman ilerle, sütun için 1)
- Shape `{2, 3, 4}` → Stride `{12, 4, 1}`

**Contiguous:** Tensor'ın bellekte kesintisiz, sıralı olarak yerleştiği durum. Transpose yapılınca stride değişir ama veri kopyalanmaz → non-contiguous olabilir. `contiguous()` çağırarak yeni, sıralı bir kopya oluşturulur.

### ONNX Nedir?

**Open Neural Network Exchange** — ML modellerini temsil eden açık format.
- PyTorch, TensorFlow gibi framework'ler modeli `.onnx` dosyasına export edebilir
- İçerik: **Protobuf** (Protocol Buffers) formatında serileştirilmiş veri
- Yapı: `ModelProto > GraphProto > NodeProto + TensorProto`
- Her node bir operatör: Conv, ReLU, MaxPool, vs.
- Weight'ler (ağırlıklar) `initializer` olarak saklanır

### Computation Graph (DAG)

Neural network aslında bir **yönlü asiklik graf (DAG)**:
- **Node'lar:** operatörler (Conv, ReLU, Add, vs.)
- **Kenarlar:** tensor'lar (verinin akışı)
- **Asiklik:** döngü yok, veri sadece ileri akar

**Topological Sort (Kahn's Algorithm):** DAG'daki node'ları "önce bağımlılıklar, sonra bağımlı olanlar" sırasına koyar.
1. In-degree'si 0 olan node'ları queue'ye ekle
2. Queue'den node çıkar, output listesine ekle
3. Bu node'un successor'larının in-degree'sini 1 azalt
4. Yeni in-degree 0 olanları queue'ye ekle
5. Queue boşalana kadar tekrarla

### Operator Registry Pattern

`std::unordered_map<std::string, std::function<Tensor(OpContext&)>>` — operatör adını fonksiyona eşler.
- `register_op("Relu", [](OpContext& ctx) { return relu(ctx.inputs[0]); })`
- `dispatch("Relu", context)` → doğru fonksiyonu çağırır
- Yeni operatör eklemek = sadece bir satır register

### SIMD (Single Instruction, Multiple Data)

CPU'nun aynı anda birden fazla veriyi işlemesi:
- **ARM NEON:** ARM64 işlemcilerde 128-bit register, 4 float aynı anda
- **SSE:** x86 işlemcilerde 128-bit register, 4 float aynı anda
- Örnek: 4 float toplama → 1 instruction (normal: 4 instruction)
- Speedup: element-wise ops'ta 9-22x

### Quantization (Niceleme)

FP32 (32-bit float) → INT8 (8-bit integer) dönüşümü:
- **Neden:** 4x daha az bellek, integer aritmetiği daha hızlı
- **Scale:** `(max - min) / 255` — float aralığını INT8'e eşler
- **Zero Point:** INT8'de sıfırın karşılığı
- **Quantize:** `q = round(x / scale) + zero_point`
- **Dequantize:** `x = (q - zero_point) * scale`
- **Tradeoff:** Hassasiyet kaybı var (INT8: ~0.125 max error)

### Operator Fusion

İki ardışık operatörü tek bir operatöre birleştirmek:
- **Conv + BatchNorm Fusion:** BN parametrelerini Conv weight'lerine gömer
  ```
  new_weight[c] = weight[c] * gamma[c] / sqrt(var[c] + eps)
  new_bias[c] = (bias[c] - mean[c]) * gamma[c] / sqrt(var[c] + eps) + beta[c]
  ```
  Sonuç: BatchNorm tamamen yok olur, sadece Conv kalır (aynı sonuç, daha hızlı)
- **Conv + ReLU Fusion:** Conv output'una hemen ReLU uygula (intermediate tensor oluşturmadan)

---

## Faz 0 — Proje Kurulumu

**Branch:** `main` (initial setup)
**Tarih:** Proje başlangıcı

### Ne Yapıldı

1. **CMake build sistemi** kuruldu:
   - C++17 standardı
   - `FetchContent` ile dependency management
   - Google Test (v1.14.0) — unit testing
   - Protocol Buffers (v25.3) — ONNX parsing
   - Abseil-cpp — protobuf'un bağımlılığı

2. **Proje yapısı** oluşturuldu:
   ```
   edge-ml-engine/
   ├── include/edge_ml/    # Header dosyaları
   ├── src/                # Kaynak kodlar
   ├── tests/              # Unit testler
   ├── benchmarks/         # Performans testleri
   ├── proto/              # ONNX protobuf şeması
   └── docs/               # Faz dokümantasyonu
   ```

3. **GitHub Actions CI/CD** pipeline'ı kuruldu:
   - Her push/PR'da otomatik build + test
   - Ubuntu runner, CMake + GTest

4. **ONNX protobuf şeması** (`proto/onnx.proto3`) eklendi — resmi ONNX protobuf tanımı (~1000 satır)

### Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `CMakeLists.txt` | Ana build konfigürasyonu |
| `.github/workflows/` | CI/CD pipeline |
| `proto/onnx.proto3` | ONNX protobuf şeması |
| `src/main.cpp` | Minimal entry point |

### Karşılaşılan Sorunlar

- **Protobuf + Abseil bağımlılığı:** Protobuf v25.3, abseil-cpp gerektiriyor. FetchContent'e abseil-cpp eklendi, `ABSL_PROPAGATE_CXX_STD=ON` ve `protobuf_ABSL_PROVIDER=package` ayarlandı.

---

## Faz 1 — Tensor Sınıfı & Bellek Motoru

**Branch:** `main` (veya `phase-1-tensor`)
**Test sayısı:** 35 test

### Ne Yapıldı

Tüm ML operasyonlarının temelini oluşturan **Tensor** sınıfı sıfırdan yazıldı.

### Tensor Sınıfı Özellikleri

**Veri yapısı:**
```cpp
class Tensor {
    std::vector<int> shape_;           // boyutlar: {2, 3, 4}
    std::vector<int> strides_;         // adımlar: {12, 4, 1}
    std::shared_ptr<float[]> data_;    // veri (paylaşımlı pointer)
    int offset_;                       // veri başlangıç offset'i
    int size_;                         // toplam eleman sayısı
};
```

**Neden `shared_ptr`?** Reshape ve transpose işlemlerinde veri kopyalamak yerine aynı belleği paylaşmak için. Zero-copy semantics.

**Constructor'lar:**
- `Tensor()` — boş tensor
- `Tensor({2, 3})` — shape ile (sıfır initialize)
- `Tensor({2, 3}, {1,2,3,4,5,6})` — shape + veri ile

**Element erişimi:**
- `tensor[i]` — düz indeks (flat buffer)
- `tensor.at({i, j, k})` — çok boyutlu indeks (stride-based)
  - `flat_index = i * stride[0] + j * stride[1] + k * stride[2] + offset`

**Shape operasyonları:**
- `reshape({3, 2})` — yeni shape, aynı veri (zero-copy, contiguous ise)
  - `-1` desteği: `reshape({-1, 3})` → eksik boyutu otomatik hesapla
- `transpose(dim0, dim1)` — iki boyutu yer değiştirir
  - Sadece stride'ları swap eder, veri kopyalamaz
  - Sonuç non-contiguous olabilir
- `contiguous()` — verinin sıralı kopyasını oluşturur
- `is_contiguous()` — stride'ların row-major olup olmadığını kontrol eder

**Element-wise operasyonlar:**
- `add(other)` — eleman bazlı toplama
- `subtract(other)` — eleman bazlı çıkarma
- `multiply(other)` — eleman bazlı çarpma
- Shape uyuşmazlığında exception fırlatır

**Matrix çarpımı:**
- `matmul(other)` — naive O(n³) implementasyon
  - 3 iç içe döngü: `C[i][j] += A[i][k] * B[k][j]`
- `matmul_tiled(other, tile_size=32)` — cache-friendly tiled versiyon
  - Matris'i tile_size × tile_size bloklara böl
  - Her bloğu işle → L1 cache'de kalır → daha hızlı

**Utility:**
- `fill(value)` — tüm elemanları aynı değere set et
- `print()` — debug output (ilk 20 eleman)
- `data()` — raw float pointer (SIMD için)

### Dosyalar

| Dosya | Satır | Açıklama |
|-------|-------|----------|
| `include/edge_ml/tensor.h` | ~80 | Tensor sınıfı tanımı |
| `src/tensor/tensor.cpp` | ~365 | Tüm implementasyonlar |
| `tests/tensor_test.cpp` | ~400 | 35 unit test |

### Test Coverage

- Constructor testleri (boş, shape, data, copy, move)
- İndeksleme testleri (flat, multi-dimensional)
- Reshape testleri (-1 inference, size mismatch hatası)
- Transpose testleri (2D, zero-copy doğrulama, contiguous dönüşüm)
- Element-wise testleri (add, sub, mul, shape mismatch hatası)
- Matmul testleri (basic, rectangular, identity, tiled)

---

## Faz 2 — Operatör Kütüphanesi

**Branch:** `phase-2-operators` → PR → merge to main
**Test sayısı:** 27 yeni test (toplam: 62)

### Ne Yapıldı

13 neural network operatörü sıfırdan implemente edildi.

### Operatörler Detaylı

#### 1. ReLU (Rectified Linear Unit)
```
f(x) = max(0, x)
```
- En basit ve en yaygın activation function
- Negatif değerleri sıfıra çeker, pozitifleri geçirir
- Gradient vanishing problemini çözer (sigmoid'e kıyasla)

#### 2. Sigmoid
```
f(x) = 1 / (1 + exp(-x))
```
- Output: [0, 1] aralığında
- Olasılık çıktısı gereken yerlerde kullanılır
- YOLO'da detection head'de confidence score

#### 3. SiLU (Sigmoid Linear Unit / Swish)
```
f(x) = x * sigmoid(x)
```
- Modern mimarilerde ReLU'nun yerini alıyor
- Self-gated: input kendi kendini kapıyor
- YOLO, EfficientNet gibi modellerde standart

#### 4. Softmax
```
f(x_i) = exp(x_i) / sum(exp(x_j))
```
- Output: tüm değerler [0,1] ve toplamları 1
- Sınıflandırma için olasılık dağılımı üretir
- **Numerically stable:** max(x) çıkarılır overflow'u önlemek için
  ```
  x_stable = x - max(x)
  softmax(x_stable)
  ```

#### 5. Conv2D (2D Convolution)

Görüntü üzerinde filtre gezdirme — CNN'lerin temel taşı.

**Naive implementasyon:** 7 iç içe döngü
```
for batch:
  for out_channel:
    for out_y:
      for out_x:
        sum = bias[out_channel]
        for in_channel:
          for kernel_y:
            for kernel_x:
              sum += input[...] * weight[...]
        output[...] = sum
```

**im2col implementasyon:** Convolution'ı matris çarpımına dönüştürür
1. Input patch'lerini sütunlara dönüştür (im2col)
2. Weight'leri satırlara dönüştür
3. `output = weight_matrix @ column_matrix`
4. Sonucu doğru shape'e reshape et

**Parametreler:**
- `stride` — filtre adım boyutu (1 = her piksel, 2 = her 2 piksel)
- `padding` — input kenarlarına sıfır ekleme (boyut koruma için)
- Input: NCHW `{batch, channels, height, width}`
- Weight: `{out_channels, in_channels, kernel_h, kernel_w}`

#### 6. Dense / Fully Connected (FC)
```
output = input @ weight^T + bias
```
- Her input nöronu her output nöronuna bağlı
- Matris çarpımı + bias ekleme
- Batched ve unbatched input destekli

#### 7. MaxPool2D
- Sliding window ile en büyük değeri seç
- Spatial boyutları küçültür (downsampling)
- Feature'ların konumsal hassasiyetini azaltır (translation invariance)
- Padding: out-of-bounds = -infinity (max'ı etkilemez)

#### 8. BatchNorm (Inference Mode)
```
output = gamma * (input - mean) / sqrt(var + eps) + beta
```
- Training'de öğrenilmiş mean/var ile normalize eder
- Inference'da precompute:
  ```
  scale[c] = gamma[c] / sqrt(var[c] + eps)
  shift[c] = beta[c] - scale[c] * mean[c]
  output = scale * input + shift
  ```
- Her channel için ayrı parametreler

#### 9-11. Flatten, Reshape, GlobalAveragePool
- **Flatten:** N-D tensor'ı 2D'ye dönüştürür (axis bazlı)
- **Reshape:** Arbitrary shape dönüşümü
- **GlobalAveragePool:** Spatial boyutları ortalayarak 1×1'e küçültür

#### 12-13. Add, Sub, Mul, Transpose
- Element-wise aritmetik operasyonlar
- Transpose: boyut permütasyonu

### Operator Registry

Singleton pattern ile operatör dispatch:
```cpp
OpRegistry& registry = OpRegistry::instance();
registry.register_op("Relu", relu_handler);
Tensor result = registry.dispatch("Relu", context);
```

`register_all_ops()` fonksiyonu tüm operatörleri tek çağrıda kaydeder.

### Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `include/edge_ml/operators/activations.h` | ReLU, Sigmoid, SiLU, Softmax tanımları |
| `src/operators/activations.cpp` | Activation implementasyonları |
| `include/edge_ml/operators/conv2d.h` | Conv2D tanımı |
| `src/operators/conv2d.cpp` | Naive + im2col implementasyon |
| `include/edge_ml/operators/dense.h` | Dense/FC tanımı |
| `src/operators/dense.cpp` | Dense implementasyon |
| `include/edge_ml/operators/pooling.h` | MaxPool2D tanımı |
| `src/operators/pooling.cpp` | Pooling implementasyon |
| `include/edge_ml/operators/batchnorm.h` | BatchNorm tanımı |
| `src/operators/batchnorm.cpp` | BatchNorm implementasyon |
| `include/edge_ml/operators/registry.h` | OpRegistry tanımı |
| `src/operators/registry.cpp` | Registry + register_all_ops |
| `tests/operators_test.cpp` | 27 unit test |

---

## Faz 3 — ONNX Parser & Graf Executor

**Branch:** `phase-3-onnx-graph` → PR → merge to main
**Test sayısı:** 9 yeni test (toplam: 71)

### Ne Yapıldı

ONNX model dosyalarını okuyup çalıştırabilen tam bir inference pipeline kuruldu.

### ONNX Parser

Protobuf ile `.onnx` dosyasını okur:

**Veri yapıları:**
```cpp
struct OnnxAttribute {
    std::string name;
    int type;              // 1=float, 2=int, 3=string, 6=floats, 7=ints
    float f;               // tek float
    int64_t i;            // tek int
    std::string s;        // string
    std::vector<float> floats;
    std::vector<int64_t> ints;
};

struct OnnxNode {
    std::string op_type;   // "Conv", "Relu", vs.
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<OnnxAttribute> attributes;
};

struct OnnxModel {
    std::vector<OnnxNode> nodes;
    std::map<std::string, Tensor> initializers;  // weight'ler
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};
```

**parse_onnx() akışı:**
1. Dosyayı binary oku
2. `ModelProto.ParseFromIstream()` ile protobuf parse
3. `graph.initializer` → weight tensor'larını yükle (FLOAT/DOUBLE/INT64)
4. `graph.input` → model input'larını bul (initializer olmayanlar)
5. `graph.output` → model output adlarını al
6. `graph.node` → her node'un op_type, inputs, outputs, attributes'ını oku

### Computation Graph

ONNX node'larından DAG oluşturur:

```cpp
struct GraphNode {
    int id;
    OnnxNode onnx_node;
    std::vector<int> predecessors;  // bu node'a giren node'lar
    std::vector<int> successors;    // bu node'dan çıkan node'lar
};
```

**build() akışı:**
1. Her node için GraphNode oluştur
2. Producer map: `tensor_name → node_id` (hangi tensor'ı hangi node üretiyor)
3. Her node'un input'larına bak → üreten node'u bul → edge ekle
4. Bidirectional: predecessor ↔ successor

**topological_sort():** Kahn's Algorithm
- O(V + E) karmaşıklık
- Cycle detection: sorted.size() < n ise döngü var

### Executor

Topological sırada operatörleri çalıştırır:

```cpp
class Executor {
    OnnxModel model_;
    Graph graph_;
    OpRegistry registry_;
    std::vector<int> execution_order_;  // topological sort sonucu
    
    Tensor run(const Tensor& input);
    Tensor run(const std::map<std::string, Tensor>& inputs);
};
```

**run() akışı:**
1. `tensor_map`'e tüm initializer'ları (weight'ler) yükle
2. `tensor_map`'e input tensor'ları yükle
3. `execution_order` sırasında her node için:
   - Node'un input'larını `tensor_map`'ten al
   - `execute_node()` çağır
   - Sonucu `tensor_map`'e yaz
4. Model output'unu döndür

**execute_node() desteklenen operasyonlar:**
- Element-wise: Add, Sub, Mul
- Activation: Relu, Sigmoid, Silu, Softmax
- Linear: MatMul, Gemm (alpha, beta, transA, transB destekli)
- Conv: Conv (im2col ile, strides/pads attribute'larından)
- Pooling: MaxPool, GlobalAveragePool
- Normalization: BatchNormalization (epsilon attribute)
- Shape: Reshape, Flatten, Transpose

### Test Model

Python ile test modeli oluşturuldu (`tests/create_test_model.py`):
```
Conv(1→4, 3×3) → ReLU → MaxPool(2×2) → Flatten → Gemm(4*3*3→10) → Softmax
```
- Input: (1, 1, 8, 8) — tek kanallı 8×8 görüntü
- Output: (1, 10) — 10 sınıflı olasılık dağılımı
- ONNXRuntime ile doğrulanmış referans output

**End-to-end test:** Engine output vs ONNXRuntime output → tolerance 1e-3 ile eşleşiyor.

### Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `include/edge_ml/parser/onnx_parser.h` | Parser veri yapıları ve fonksiyon tanımları |
| `src/parser/onnx_parser.cpp` | Protobuf parsing implementasyonu |
| `include/edge_ml/graph/graph.h` | Graph sınıfı ve GraphNode tanımı |
| `src/graph/graph.cpp` | DAG build + topological sort |
| `include/edge_ml/graph/executor.h` | Executor sınıfı tanımı |
| `src/graph/executor.cpp` | Tüm operatörlerin execution logic'i |
| `tests/graph_test.cpp` | 9 test (parse, graph, executor, end-to-end) |
| `tests/create_test_model.py` | Test ONNX modeli oluşturan Python scripti |
| `tests/test_model.onnx` | Oluşturulan test modeli (~800 bytes) |

### Karşılaşılan Sorunlar

- **Test dosya yolu:** Build dizininden test_model.onnx'e erişim sorunu. CMake `target_compile_definitions` ile `TEST_DATA_DIR` tanımlandı.
- **Python bağımlılıkları:** `pip3 install onnx numpy onnxruntime` gerekti.

---

## Faz 4 — SIMD, Quantization, Operator Fusion

**Branch:** `phase-4-optimization` → PR → merge to main
**Test sayısı:** 10 yeni test (toplam: 81)

### Ne Yapıldı

Üç optimizasyon katmanı eklendi: SIMD vektörizasyon, quantization, ve operator fusion.

### SIMD Operasyonları

Conditional compilation ile platform desteği:
```cpp
#if defined(__ARM_NEON)    // Apple Silicon, ARM64
    #include <arm_neon.h>
#elif defined(__SSE__)      // Intel/AMD x86
    #include <xmmintrin.h>
#endif
```

**simd_add(a, b):**
```cpp
// 4 float'u aynı anda topla
float32x4_t va = vld1q_f32(src_a + i);   // 4 float yükle
float32x4_t vb = vld1q_f32(src_b + i);   // 4 float yükle
float32x4_t vr = vaddq_f32(va, vb);       // 4 toplama, 1 instruction
vst1q_f32(dst + i, vr);                   // 4 float yaz
// Kalan elemanlar scalar fallback ile
```

**simd_matmul(a, b):** Tiled + SIMD
- 32×32 tile boyutu (L1 cache'e sığacak şekilde)
- Her tile içinde NEON/SSE ile 4-wide FMA (fused multiply-add)
- `vmlaq_f32(vc, va, vb)` = vc += va * vb (tek instruction)

**Benchmark sonuçları (Apple Silicon):**

| Operasyon | Naive | SIMD | Speedup |
|-----------|-------|------|---------|
| Add (1M) | 34.9 ms | 1.6 ms | **21x** |
| Multiply (1M) | 34.5 ms | 1.6 ms | **22x** |
| ReLU (1M) | 11.6 ms | 1.2 ms | **9.5x** |

### Conv + BatchNorm Fusion

BN parametrelerini Conv weight'lerine gömer, BN node'unu graftan siler:

**Matematik:**
```
// Orijinal: conv_out = W * x + b, bn_out = gamma * (conv_out - mean) / sqrt(var + eps) + beta
// Birleşik: fused_out = W_new * x + b_new

scale[c] = gamma[c] / sqrt(var[c] + eps)
W_new[c] = W[c] * scale[c]                              // tüm weight'ler scale ile çarp
b_new[c] = (b[c] - mean[c]) * scale[c] + beta[c]       // bias'ı güncelle
```

**fuse_graph() fonksiyonu:**
1. **Pass 1:** Conv + BatchNorm çiftlerini bul → weight'leri birleştir → BN'i sil
2. **Pass 2:** Conv + ReLU çiftlerini bul → Conv'a `fused_relu` attribute ekle → ReLU'yu sil
3. **Pass 3:** Boş node'ları graftan temizle

### INT8 Quantization

**quantize_int8():**
1. Min/max bul
2. `scale = (max - min) / 255`
3. `zero_point = -128 - min / scale` (clipped)
4. Her eleman: `q[i] = clamp(round(x[i] / scale) + zp, -128, 127)`

**quantized_matmul():**
1. A ve B'yi INT8'e quantize et
2. INT8 çarpma, INT32 accumulation (overflow koruması)
3. Sonucu FP32'ye dequantize et: `out = acc * (scale_a * scale_b)`

**Sıkıştırma:**
- FP32: 4 byte/eleman → INT8: 1 byte/eleman → **4x bellek tasarrufu**
- 1M parametre: 3906 KB → 976 KB

### FP16 Simulation

FP32 mantissa'sının alt 13 bitini sıfırlayarak FP16 hassasiyet kaybını simüle eder:
```cpp
uint32_t bits;
memcpy(&bits, &src[i], sizeof(float));
bits &= 0xFFFFE000u;  // 23-bit mantissa → 10-bit (FP16)
memcpy(&dst[i], &bits, sizeof(float));
```
Max hassasiyet kaybı: ~0.0005

### Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `include/edge_ml/optimizer/simd_ops.h` | SIMD fonksiyon tanımları |
| `src/optimizer/simd_ops.cpp` | NEON + SSE implementasyon |
| `include/edge_ml/optimizer/fusion.h` | Fusion fonksiyon tanımları |
| `src/optimizer/fusion.cpp` | Conv+BN ve Conv+ReLU fusion |
| `include/edge_ml/optimizer/quantization.h` | Quantization veri yapıları + fonksiyonlar |
| `src/optimizer/quantization.cpp` | INT8 quant/dequant, quantized matmul, FP16 sim |
| `tests/optimization_test.cpp` | 10 test |
| `benchmarks/matmul_bench.cpp` | Matmul benchmark (naive vs tiled) |
| `benchmarks/optimization_bench.cpp` | SIMD + quant + fusion benchmark suite |

### Karşılaşılan Sorunlar

- **CI build hatası:** `matmul_bench.cpp`'de `#include <functional>` eksikti. macOS Clang otomatik dahil ediyor ama Linux GCC etmiyor. Include eklenerek çözüldü.

---

## Mevcut Proje Özeti

### Sayılar

| Metrik | Değer |
|--------|-------|
| Toplam test | 81 (hepsi geçiyor) |
| Operatör sayısı | 13 |
| Kaynak dosya | ~14 .cpp dosyası |
| Header dosya | ~12 .h dosyası |
| Satır kodu (tahmini) | ~3000+ |
| CI/CD | GitHub Actions (her push'ta) |

### Proje Ağacı

```
edge-ml-engine/
├── CMakeLists.txt
├── README.md
├── include/edge_ml/
│   ├── tensor.h
│   ├── operators/
│   │   ├── activations.h      (ReLU, Sigmoid, SiLU, Softmax)
│   │   ├── conv2d.h           (Conv2D naive + im2col)
│   │   ├── dense.h            (Fully Connected)
│   │   ├── pooling.h          (MaxPool2D)
│   │   ├── batchnorm.h        (Batch Normalization)
│   │   └── registry.h         (Operator Registry)
│   ├── parser/
│   │   └── onnx_parser.h      (ONNX Protobuf Parser)
│   ├── graph/
│   │   ├── graph.h            (DAG + Topological Sort)
│   │   └── executor.h         (Sequential Executor)
│   └── optimizer/
│       ├── simd_ops.h         (NEON/SSE SIMD)
│       ├── fusion.h           (Conv+BN, Conv+ReLU Fusion)
│       └── quantization.h     (INT8, FP16)
├── src/
│   ├── main.cpp
│   ├── tensor/tensor.cpp
│   ├── operators/
│   │   ├── activations.cpp
│   │   ├── conv2d.cpp
│   │   ├── dense.cpp
│   │   ├── pooling.cpp
│   │   ├── batchnorm.cpp
│   │   └── registry.cpp
│   ├── parser/onnx_parser.cpp
│   ├── graph/
│   │   ├── graph.cpp
│   │   └── executor.cpp
│   └── optimizer/
│       ├── simd_ops.cpp
│       ├── fusion.cpp
│       └── quantization.cpp
├── tests/
│   ├── tensor_test.cpp        (35 test)
│   ├── operators_test.cpp     (27 test)
│   ├── graph_test.cpp         (9 test)
│   ├── optimization_test.cpp  (10 test)
│   ├── create_test_model.py
│   └── test_model.onnx
├── benchmarks/
│   ├── matmul_bench.cpp
│   └── optimization_bench.cpp
├── proto/
│   └── onnx.proto3
└── docs/
    ├── faz-0-kurulum.md
    ├── faz-1-tensor.md
    ├── faz-2-operatorler.md
    ├── faz-3-onnx-graf.md
    └── faz-4-optimizasyon.md
```

---

## Sıradaki: Faz 5-8

> Detaylı plan için: [`nextphases.md`](nextphases.md)

| Faz | Hedef | Durum |
|-----|-------|-------|
| Faz 5 | 12 yeni operatör (YOLO için gerekli) | Planlandı |
| Faz 6 | YOLO pipeline: export → inference → NMS | Planlandı |
| Faz 7 | Multi-thread, memory pool, profiling | Planlandı |
| Faz 8 | Demo, benchmark, portfolio | Planlandı |
