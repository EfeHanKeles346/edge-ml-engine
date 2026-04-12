# Before Next Steps — Faz 5'e Girmeden Önce Yapılacak Düzeltmeler

> Bu dosya, Faz 5'e (yeni operatörler) geçmeden önce mevcut kod tabanındaki
> kritik zayıflıkları ve teknik borçları listeler. Bunlar düzeltilmeden devam
> edilirse YOLO pipeline'ında büyük sorunlarla karşılaşılacak.

---

## Düzeltme 1 — Executor: Multi-Output Node Desteği

### Sorun

Şu anda `executor.cpp` her node'un **sadece tek output** ürettiğini varsayıyor:

```cpp
// executor.cpp — mevcut hali
Tensor result = execute_node(node, tensor_map);
if (!node.outputs.empty()) {
    tensor_map[node.outputs[0]] = result;  // Sadece ilk output!
}
```

**Split** operatörü birden fazla output üretir. Örnek:
```
Split(input, axis=1, split=[32, 32])
  → output_0: {1, 32, 80, 80}
  → output_1: {1, 32, 80, 80}
```

Mevcut executor bunu handle edemez. YOLO'da Split **~10+ kez** kullanılıyor.

### Çözüm

`execute_node()` dönüş tipini değiştir: `Tensor` → `std::vector<Tensor>`

```cpp
// Yeni imza
std::vector<Tensor> execute_node(const OnnxNode& node,
                                  std::map<std::string, Tensor>& tensor_map);

// Yeni çağrı
std::vector<Tensor> results = execute_node(node, tensor_map);
for (size_t i = 0; i < results.size() && i < node.outputs.size(); i++) {
    tensor_map[node.outputs[i]] = results[i];
}
```

Mevcut tek-output operatörler için: `return {result};` (tek elemanlı vector)

### Etkilenen Dosyalar

| Dosya | Değişiklik |
|-------|-----------|
| `include/edge_ml/graph/executor.h` | `execute_node` dönüş tipi değişikliği |
| `src/graph/executor.cpp` | Tüm `execute_node` içindeki return'lar `{result}` olacak |
| `src/graph/executor.cpp` | `run()` içindeki çağrı multi-output uyumlu olacak |

### Test

```cpp
// Mevcut testler hala geçmeli (tek output geriye uyumlu)
// Yeni test: multi-output node mock
TEST(ExecutorTest, MultiOutputNode) {
    // Split benzeri bir node oluştur
    // İki output'un da tensor_map'e yazıldığını doğrula
}
```

---

## Düzeltme 2 — Tensor: Bounds Checking

### Sorun

`flat_index()` ve `at()` fonksiyonları sınır kontrolü yapmıyor:

```cpp
// tensor.cpp — mevcut hali
int Tensor::flat_index(const std::vector<int>& indices) const {
    int idx = offset_;
    for (int i = 0; i < ndim(); i++) {
        idx += indices[i] * strides_[i];  // indices[i] negatif veya > shape[i] olabilir!
    }
    return idx;
}
```

**Risk:** Out-of-bounds memory erişimi → undefined behavior, crash, veya sessiz yanlış sonuç.

### Çözüm

```cpp
int Tensor::flat_index(const std::vector<int>& indices) const {
    if (static_cast<int>(indices.size()) != ndim()) {
        throw std::invalid_argument(
            "flat_index: expected " + std::to_string(ndim()) +
            " indices, got " + std::to_string(indices.size()));
    }
    int idx = offset_;
    for (int i = 0; i < ndim(); i++) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::out_of_range(
                "flat_index: index " + std::to_string(indices[i]) +
                " out of range for dimension " + std::to_string(i) +
                " with size " + std::to_string(shape_[i]));
        }
        idx += indices[i] * strides_[i];
    }
    return idx;
}
```

Ayrıca `operator[]` için de basit bounds check:

```cpp
float& Tensor::operator[](int index) {
    if (index < 0 || index >= size_) {
        throw std::out_of_range(
            "Tensor[]: index " + std::to_string(index) +
            " out of range for size " + std::to_string(size_));
    }
    return data_[offset_ + index];
}
```

> **Not:** Performans kritik yerlerde (SIMD inner loop gibi) `data()` raw pointer
> ile erişim yapılıyor — oralar zaten güvenli çünkü döngü sınırları kontrollü.
> Bounds checking sadece public API'da (at, operator[]) olmalı.

### Etkilenen Dosyalar

| Dosya | Değişiklik |
|-------|-----------|
| `include/edge_ml/tensor.h` | Değişiklik yok (imzalar aynı) |
| `src/tensor/tensor.cpp` | `flat_index()`, `at()`, `operator[]` güncelleme |

### Test

```cpp
TEST(TensorTest, BoundsCheckAt) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(t.at({2, 0}), std::out_of_range);   // row out of range
    EXPECT_THROW(t.at({0, 3}), std::out_of_range);   // col out of range
    EXPECT_THROW(t.at({-1, 0}), std::out_of_range);  // negatif index
    EXPECT_THROW(t.at({0}), std::invalid_argument);   // eksik boyut
    EXPECT_NO_THROW(t.at({1, 2}));                    // geçerli
}

TEST(TensorTest, BoundsCheckOperator) {
    Tensor t({3}, {1, 2, 3});
    EXPECT_THROW(t[3], std::out_of_range);
    EXPECT_THROW(t[-1], std::out_of_range);
    EXPECT_NO_THROW(t[2]);
}
```

---

## Düzeltme 3 — Tensor: Integer Overflow Koruması

### Sorun

Tensor constructor'da `size_` hesaplanırken overflow olabilir:

```cpp
// tensor.cpp — mevcut hali
size_ = 1;
for (int d : shape_) size_ *= d;
// shape = {100000, 100000} → 10^10 → int overflow (max ~2.1×10^9)
```

Overflow sonucu `size_` küçük pozitif bir sayı olur → küçük buffer ayrılır → sonra büyük indeksle erişim → crash.

### Çözüm

```cpp
Tensor::Tensor(const std::vector<int>& shape) : shape_(shape), offset_(0) {
    // Shape validation
    for (int d : shape_) {
        if (d < 0) {
            throw std::invalid_argument(
                "Tensor: negative dimension " + std::to_string(d));
        }
    }

    // Safe size calculation with overflow check
    int64_t total = 1;
    for (int d : shape_) {
        total *= static_cast<int64_t>(d);
        if (total > static_cast<int64_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "Tensor: total size exceeds int range (" +
                std::to_string(total) + " elements)");
        }
    }
    size_ = static_cast<int>(total);

    compute_strides();
    data_ = std::shared_ptr<float[]>(new float[size_]());
}
```

### Etkilenen Dosyalar

| Dosya | Değişiklik |
|-------|-----------|
| `src/tensor/tensor.cpp` | Tüm constructor'lar (shape alan her biri) |

### Test

```cpp
TEST(TensorTest, NegativeDimension) {
    EXPECT_THROW(Tensor({-1, 3}), std::invalid_argument);
    EXPECT_THROW(Tensor({2, -5}), std::invalid_argument);
}

TEST(TensorTest, OverflowProtection) {
    // 100000 * 100000 = 10^10 > INT_MAX
    EXPECT_THROW(Tensor({100000, 100000}), std::overflow_error);
}

TEST(TensorTest, ZeroDimension) {
    // Bu davranış tanımlanmalı — ya izin ver ya hata ver
    // Önerim: izin ver, size_ = 0
    Tensor t({0});
    EXPECT_EQ(t.size(), 0);
}
```

---

## Düzeltme 4 — CMake: Compiler Warnings

### Sorun

CMakeLists.txt'de hiçbir warning flag'i yok. Potansiyel bug'lar (unused variables,
implicit conversions, sign comparison) sessizce derleniyor.

### Çözüm

CMakeLists.txt'e eklenecek:

```cmake
# Compiler warnings — hataları derleme aşamasında yakala
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(edge_ml PRIVATE
        -Wall           # Temel uyarılar
        -Wextra         # Ekstra uyarılar
        -Wpedantic      # Standart uyumluluğu
        -Wno-unused-parameter  # Bazı callback'lerde unused parameter normal
    )
endif()

# Debug build'de Address Sanitizer (opsiyonel ama çok faydalı)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(edge_ml PRIVATE -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(edge_ml PRIVATE -fsanitize=address)
    endif()
endif()
```

### Sonra Yapılacak

1. Warning'lerle derle
2. Çıkan uyarıları tek tek düzelt
3. Tipik uyarılar:
   - `int` vs `size_t` comparison (signed/unsigned mismatch)
   - Unused variables
   - Implicit float/double conversions
   - Missing return statements

### Etkilenen Dosyalar

| Dosya | Değişiklik |
|-------|-----------|
| `CMakeLists.txt` | Warning flags ekleme |
| Çeşitli `.cpp` dosyaları | Uyarıları düzeltme (cast ekleme, unused kaldırma vs.) |

---

## Düzeltme 5 — Broadcasting Desteği (Temel Seviye)

### Sorun

Mevcut element-wise operasyonlar (add, subtract, multiply) sadece **aynı shape** ile çalışıyor:

```cpp
// tensor.cpp — mevcut hali
Tensor Tensor::add(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shape mismatch for add");
    }
    // ...
}
```

YOLO'da şu tür işlemler var:
```
{1, 64, 80, 80} + {1, 64, 1, 1}   → bias broadcast
{1, 80} * {1, 1}                    → scalar broadcast
{8400, 4} + {1, 4}                  → row broadcast
```

Broadcasting olmadan bunlar **çalışmaz**.

### Çözüm: NumPy-style Broadcasting (Temel)

**Broadcasting kuralları:**
1. Shape'leri sağdan hizala
2. Her boyut için: ya eşit, ya biri 1
3. Boyutu 1 olan → diğerine genişletilir

```
{1, 64, 80, 80}  ← 4D
    {1, 64, 1, 1}  ← 4D
= {1, 64, 80, 80}  ← sonuç

{8400, 4}  ← 2D
    {1, 4}  ← 2D
= {8400, 4}  ← sonuç
```

**Implementasyon:**

```cpp
// Yeni helper fonksiyon
std::vector<int> broadcast_shape(const std::vector<int>& a, const std::vector<int>& b);

// Güncellenmiş add
Tensor Tensor::add(const Tensor& other) const {
    if (shape_ == other.shape_) {
        // Fast path: aynı shape, mevcut kod
        return add_same_shape(other);
    }

    // Broadcast path
    auto out_shape = broadcast_shape(shape_, other.shape_);
    Tensor output(out_shape);

    // Broadcast index hesaplama ile element-wise toplama
    // Her output indeksi için kaynak indekslerini hesapla
    // (boyutu 1 olan dimension'da index her zaman 0)
    for (int i = 0; i < output.size(); i++) {
        auto out_idx = output.unravel_index(i);
        auto a_idx = broadcast_index(out_idx, shape_);
        auto b_idx = broadcast_index(out_idx, other.shape_);
        output[i] = this->at(a_idx) + other.at(b_idx);
    }
    return output;
}
```

**Yeni yardımcı fonksiyonlar:**
```cpp
// Flat index → multi-dimensional index
std::vector<int> unravel_index(int flat_idx) const;

// Broadcast'te kaynak index: boyutu 1 olan dimension → index 0
std::vector<int> broadcast_index(const std::vector<int>& out_idx,
                                  const std::vector<int>& src_shape);
```

### Etkilenen Dosyalar

| Dosya | Değişiklik |
|-------|-----------|
| `include/edge_ml/tensor.h` | `broadcast_shape()`, `unravel_index()`, `broadcast_index()` tanımları |
| `src/tensor/tensor.cpp` | `add()`, `subtract()`, `multiply()` güncelleme + broadcast helpers |

### Test

```cpp
TEST(TensorTest, BroadcastScalar) {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({1}, {10});
    Tensor result = a.add(b);
    EXPECT_EQ(result.shape(), (std::vector<int>{2, 3}));
    EXPECT_FLOAT_EQ(result[0], 11.0f);
    EXPECT_FLOAT_EQ(result[5], 16.0f);
}

TEST(TensorTest, BroadcastRow) {
    Tensor a({3, 4}, std::vector<float>(12, 1.0f));
    Tensor b({1, 4}, {10, 20, 30, 40});
    Tensor result = a.add(b);
    EXPECT_EQ(result.shape(), (std::vector<int>{3, 4}));
    EXPECT_FLOAT_EQ(result.at({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(result.at({2, 3}), 41.0f);
}

TEST(TensorTest, BroadcastBias4D) {
    // YOLO'daki tipik bias broadcast
    Tensor a({1, 2, 3, 3}, std::vector<float>(18, 1.0f));
    Tensor b({1, 2, 1, 1}, {10.0f, 20.0f});
    Tensor result = a.add(b);
    EXPECT_EQ(result.shape(), (std::vector<int>{1, 2, 3, 3}));
    EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0}), 11.0f);  // channel 0: +10
    EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0}), 21.0f);  // channel 1: +20
}

TEST(TensorTest, BroadcastIncompatible) {
    Tensor a({2, 3});
    Tensor b({2, 4});
    EXPECT_THROW(a.add(b), std::invalid_argument);  // 3 != 4, ikisi de != 1
}
```

---

## Uygulama Sırası

| Sıra | Düzeltme | Tahmini Süre | Bağımlılık |
|------|----------|-------------|------------|
| 1 | CMake warnings (Düzeltme 4) | 30 dk | Yok — önce yap, diğer düzeltmelerdeki uyarıları yakalar |
| 2 | Integer overflow (Düzeltme 3) | 1 saat | Yok |
| 3 | Bounds checking (Düzeltme 2) | 1 saat | Düzeltme 3 (overflow fix'ten sonra) |
| 4 | Broadcasting (Düzeltme 5) | 2-3 saat | Düzeltme 2 (bounds check'li at() lazım) |
| 5 | Multi-output executor (Düzeltme 1) | 2-3 saat | Yok, ama en son yap — en çok dosyayı etkiliyor |

**Toplam tahmini süre: 1-2 gün**

---

## Tamamlanma Kontrol Listesi

- [ ] **Düzeltme 1:** Executor multi-output destekliyor
- [ ] **Düzeltme 2:** flat_index() ve operator[] bounds checking yapıyor
- [ ] **Düzeltme 3:** Tensor constructor integer overflow'a karşı korumalı
- [ ] **Düzeltme 4:** CMake -Wall -Wextra aktif, tüm uyarılar temizlenmiş
- [ ] **Düzeltme 5:** add/subtract/multiply broadcasting destekliyor
- [ ] Tüm mevcut 81 test hala geçiyor (geriye uyumluluk)
- [ ] Yeni düzeltmeler için ~15-20 yeni test eklendi
- [ ] Branch: `before-phase5-fixes` → PR → merge to main
- [ ] memory.md güncellendi (yapılan düzeltmeler eklendi)

---

## Bu Düzeltmeler Sonrası

Engine Faz 5'e **sağlam bir temelle** girecek:
- ✅ Multi-output ops (Split) çalışabilecek
- ✅ Memory safety (bounds check + overflow protection)
- ✅ Broadcasting (YOLO bias/scale işlemleri)
- ✅ Compiler uyarıları temizlenmiş

**Sonraki adım → `nextphases.md` Faz 5: Eksik operatörler**
