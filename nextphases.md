# Edge ML Engine — v2 Roadmap (Faz 5 - Faz 8)

> **Hedef:** YOLOv11-nano modelini sıfırdan yazılmış C++ inference engine üzerinde çalıştırmak.
> **Mevcut Durum:** v1 (Faz 0-4) tamamlandı — tensor, 13 operatör, ONNX parser, graf executor, SIMD, quantization, fusion.

---

## Faz 5 — Eksik Operatörler

YOLOv11-nano ONNX modeli ~25 farklı operatör tipi kullanıyor. Bunlardan 13'ü zaten var.
Bu fazda eksik olan **12 operatör** implement edilecek.

### Öncelik Sırası ve Detaylar

#### Step 5.1 — Concat (KRİTİK)

**Ne yapıyor:** Birden fazla tensor'ı belirli bir axis boyunca birleştirir.
**YOLO'da nerede:** FPN/PAN neck'te farklı scale'lerin birleştirilmesi, C3k2 blokları içinde. ~20+ kez kullanılıyor.
**ONNX adı:** `Concat`
**Attribute:** `axis` (int, zorunlu)

**Implementasyon:**
1. Header: `include/edge_ml/operators/concat.h`
2. Source: `src/operators/concat.cpp`
3. Fonksiyon imzası: `Tensor concat(const std::vector<Tensor>& inputs, int axis)`
4. Algoritma:
   - Tüm input'ların axis dışındaki boyutlarının eşit olduğunu doğrula
   - Output shape: axis boyutunda toplam = tüm input'ların axis boyutlarının toplamı
   - Output tensor'ı oluştur
   - Her input'ı sırayla output'a kopyala (offset'i artırarak)
5. Test: farklı axis'lerde concat, 2D/3D/4D tensor'lar, tek tensor concat
6. Registry'ye kayıt: `"Concat"` → concat fonksiyonu
7. Executor'a ekleme: birden fazla input'u toplayıp concat'e ver

**Örnek:**
```
A = [1, 2, 3]  shape: (1, 3)
B = [4, 5, 6]  shape: (1, 3)
concat([A, B], axis=0) → [[1,2,3],[4,5,6]]  shape: (2, 3)
concat([A, B], axis=1) → [[1,2,3,4,5,6]]    shape: (1, 6)
```

---

#### Step 5.2 — Resize / Upsample (KRİTİK)

**Ne yapıyor:** Tensor'ın spatial boyutlarını büyütür veya küçültür.
**YOLO'da nerede:** Neck'te upsampling (2x nearest). 2-3 kez kullanılıyor.
**ONNX adı:** `Resize`
**Attribute:** `mode` ("nearest" veya "linear"), `coordinate_transformation_mode`

**Implementasyon:**
1. Header: `include/edge_ml/operators/resize.h`
2. Source: `src/operators/resize.cpp`
3. Fonksiyon imzası: `Tensor resize(const Tensor& input, const std::vector<float>& scales, const std::string& mode)`
4. İki mod:
   - **Nearest:** Her output piksel için en yakın input pikseli seç
     ```
     src_y = floor(dst_y / scale_h)
     src_x = floor(dst_x / scale_w)
     output[n][c][dst_y][dst_x] = input[n][c][src_y][src_x]
     ```
   - **Bilinear:** 4 komşu pikselin ağırlıklı ortalaması (şimdilik opsiyonel, YOLO nearest kullanıyor)
5. Input format: NCHW (batch, channels, height, width)
6. Output shape: `{N, C, H*scale_h, W*scale_w}`
7. Test: 2x nearest upsample, farklı scale'ler, bilinear (opsiyonel)
8. Registry + Executor ekleme

**YOLO kullanımı:** Genelde `scales = [1, 1, 2, 2]` (batch ve channel sabit, H ve W 2 katına)

---

#### Step 5.3 — Split (KRİTİK)

**Ne yapıyor:** Bir tensor'ı belirli axis boyunca parçalara böler.
**YOLO'da nerede:** C3k2 channel splitting, detection head'de bbox/class split. ~10+ kez.
**ONNX adı:** `Split`
**Attribute:** `axis` (int)
**Input:** data tensor + split sizes tensor

**Implementasyon:**
1. Header: `include/edge_ml/operators/split.h`
2. Source: `src/operators/split.cpp`
3. Fonksiyon imzası: `std::vector<Tensor> split(const Tensor& input, const std::vector<int>& split_sizes, int axis)`
4. Algoritma:
   - Her parça için output tensor oluştur
   - Input'tan ilgili slice'ları kopyala
   - Concat'in tersi mantık
5. Test: eşit bölme, farklı boyutlarda bölme, farklı axis'ler
6. Registry + Executor: **Dikkat** — Split birden fazla output üretiyor, executor'da özel handling lazım

**Executor değişikliği:** Şu an `execute_node` tek tensor dönüyor. Split için `std::vector<Tensor>` dönecek şekilde güncellenmeli, veya her output'u ayrı ayrı tensor_map'e yazmalı.

---

#### Step 5.4 — Slice (KRİTİK)

**Ne yapıyor:** Tensor'dan bir alt bölge çıkarır (Python'daki `tensor[2:5, :, 1:3]` gibi).
**YOLO'da nerede:** Sub-tensor extraction, Split'in internal implementasyonu. ~10+ kez.
**ONNX adı:** `Slice`
**Input:** data, starts, ends, axes, steps (hepsi tensor olarak)

**Implementasyon:**
1. Header: `include/edge_ml/operators/slice.h`
2. Source: `src/operators/slice.cpp`
3. Fonksiyon imzası: `Tensor slice(const Tensor& input, const std::vector<int>& starts, const std::vector<int>& ends, const std::vector<int>& axes, const std::vector<int>& steps)`
4. Algoritma:
   - Her axis için start:end:step range hesapla
   - Negatif indeks desteği (Python-style: -1 = son eleman)
   - Output shape hesapla
   - Nested loop ile elemanları kopyala
5. Test: tek axis slice, multi-axis, negatif indeks, step > 1
6. Registry + Executor

---

#### Step 5.5 — Div (YÜKSEK)

**Ne yapıyor:** Element-wise bölme.
**YOLO'da nerede:** Normalization, scaling işlemleri.
**ONNX adı:** `Div`

**Implementasyon:**
1. Tensor sınıfına `divide()` metodu ekle (add/subtract/multiply ile aynı pattern)
2. Sıfıra bölme kontrolü
3. Registry + Executor'a ekle
4. Test

---

#### Step 5.6 — Shape, Constant, ConstantOfShape (YÜKSEK)

**Utility operatörler** — basit ama ONNX grafında sık kullanılıyor.

**Shape:**
- Input tensor'ın shape'ini 1D INT64 tensor olarak döner
- `Tensor shape_op(const Tensor& input)` → `{dim0, dim1, dim2, ...}`

**Constant:**
- ONNX grafında sabit değer tutan node
- Attribute'dan `value` tensor'ını oku ve output olarak ver
- Executor'da: attribute'dan tensor oluştur

**ConstantOfShape:**
- Verilen shape'te, verilen sabit değerle dolu tensor oluştur
- `Tensor constant_of_shape(const std::vector<int>& shape, float value)`

---

#### Step 5.7 — Unsqueeze ve Squeeze (YÜKSEK)

**Unsqueeze:** Belirli axis'lere boyut=1 ekler. `{3, 4}` + axis=0 → `{1, 3, 4}`
**Squeeze:** Boyut=1 olan axis'leri kaldırır. `{1, 3, 1, 4}` + axes={0,2} → `{3, 4}`

**Implementasyon:** Esasen reshape wrapper'ları. Yeni shape hesapla, `reshape()` çağır.

---

#### Step 5.8 — Gather (YÜKSEK)

**Ne yapıyor:** Index tensor kullanarak bir axis boyunca değer toplar.
**YOLO'da nerede:** DFL (Distribution Focal Loss) bbox decoding.
**ONNX adı:** `Gather`
**Attribute:** `axis` (int, default 0)

**Implementasyon:**
1. Fonksiyon: `Tensor gather(const Tensor& data, const Tensor& indices, int axis)`
2. Output shape: data shape'inde axis boyutu indices shape ile değiştirilir
3. Her index pozisyonunda `data[..., indices[i], ...]` al
4. Test: 1D gather, 2D gather farklı axis'ler

---

#### Step 5.9 — ReduceSum (ORTA)

**Ne yapıyor:** Belirli axis(ler) boyunca toplam.
**YOLO'da nerede:** DFL expected value hesaplama.
**ONNX adı:** `ReduceSum`
**Attribute:** `axes`, `keepdims`

**Implementasyon:**
1. Fonksiyon: `Tensor reduce_sum(const Tensor& input, const std::vector<int>& axes, bool keepdims)`
2. Belirtilen axis'ler boyunca elemanları topla
3. `keepdims=true` ise output'ta o axis'ler 1 olarak kalır
4. Test: tek axis, multi-axis, keepdims true/false

---

#### Step 5.10 — Exp (ORTA)

**Ne yapıyor:** Element-wise e^x.
**Implementasyon:** `std::exp()` ile basit loop. Registry + test.

---

#### Step 5.11 — Pad (DÜŞÜK)

**Ne yapıyor:** Tensor kenarlarına padding ekler.
**Not:** Çoğu padding Conv operatöründe zaten handle ediliyor. ONNX simplifier çalıştırılırsa ortadan kalkabilir.
**Implementasyon:** `mode` attribute'una göre constant/reflect/edge padding.

---

#### Step 5.12 — Clip (DÜŞÜK)

**Ne yapıyor:** Değerleri [min, max] aralığına sıkıştırır.
**Implementasyon:** Basit `std::max(min_val, std::min(max_val, x))` loop.

---

### Faz 5 Tamamlanma Kriterleri

- [ ] 12 yeni operatör implement edildi
- [ ] Her operatör için en az 3 unit test yazıldı
- [ ] Tüm operatörler registry'ye kayıt edildi
- [ ] Executor güncellendi (özellikle multi-output ops: Split)
- [ ] Tüm testler geçiyor (81 mevcut + ~40 yeni = ~120+ test)
- [ ] Branch: `phase-5-operators` → PR → merge to main

---

## Faz 6 — YOLO Pipeline

Bu faz gerçek bir YOLO modelini engine üzerinde çalıştırmayı hedefliyor.

### Step 6.1 — YOLO Model Export

1. Python'da ultralytics ile YOLOv11-nano export:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolo11n.pt")
   model.export(format="onnx", simplify=True, opset=13)
   ```
2. ONNX Simplifier çalıştır (opsiyonel):
   ```python
   import onnxsim
   model_sim, check = onnxsim.simplify(model)
   ```
3. Model'deki operatörleri listele ve Faz 5'te hepsini karşıladığımızı doğrula:
   ```python
   import onnx
   model = onnx.load("yolo11n.onnx")
   ops = sorted(set(n.op_type for n in model.graph.node))
   print(ops)
   ```
4. Export edilen modeli `models/yolo11n.onnx` olarak projeye ekle

### Step 6.2 — Model Parse & Doğrulama

1. `onnx_parser` ile modeli yükle
2. Her node'un op_type'ını kontrol et — hepsi registry'de kayıtlı mı?
3. Kayıtlı olmayan operatör varsa Faz 5'e geri dön
4. Graf build et, topological sort çalıştır
5. Node sayısı, weight sayısı, input/output shape'lerini yazdır
6. Test: `tests/yolo_test.cpp` — model parsing testi

### Step 6.3 — Image Preprocessing

YOLO'ya girmeden önce görüntü hazırlanmalı:

1. **Image loading:** stb_image veya basit PPM/BMP reader (bağımlılık minimizasyonu)
2. **Resize:** Orijinal görüntüyü 640x640'a resize et
   - Aspect ratio koruma (letterbox): uzun kenar 640, kısa kenar pad
   - Pad değeri: 114/255 = 0.447 (gri)
3. **Normalize:** Piksel değerlerini [0, 1] aralığına getir (/ 255.0)
4. **Format dönüşümü:** HWC (height, width, channels) → CHW (channels, height, width)
5. **Batch boyutu ekle:** CHW → NCHW: `{1, 3, 640, 640}`

**Dosyalar:**
- `include/edge_ml/vision/preprocessing.h`
- `src/vision/preprocessing.cpp`

**Fonksiyonlar:**
```cpp
Tensor load_image(const std::string& path);           // dosyadan yükle
Tensor resize_image(const Tensor& img, int h, int w); // bilinear resize
Tensor letterbox(const Tensor& img, int target_size);  // aspect ratio koruyarak resize + pad
Tensor normalize(const Tensor& img);                   // /255.0
Tensor hwc_to_chw(const Tensor& img);                 // kanal sırası değiştir
Tensor preprocess(const std::string& path);            // hepsini tek çağrıda
```

### Step 6.4 — End-to-End Inference

1. Preprocessing → Tensor (1, 3, 640, 640)
2. Executor.run(input_tensor) → output Tensor
3. Output shape doğrulama: (1, 84, 8400) bekleniyor

**Test sırası:**
- Önce küçük test modeli ile (mevcut test_model.onnx — çalışıyor)
- Sonra YOLO'nun ilk birkaç layer'ı ile (kısmi model)
- Son olarak tam YOLO modeli ile

**Debugging stratejisi:**
- Her layer'ın output'unu Python ONNXRuntime ile karşılaştır
- Layer-by-layer doğrulama scripti yaz (`tools/validate_layer.py`)
- Tolerance: 1e-3 (FP32 karşılaştırma)

### Step 6.5 — Output Decode (Bbox)

YOLO output'u: `(1, 84, 8400)` → post-processing gerekiyor.

**Adımlar:**
1. **Transpose:** (1, 84, 8400) → (1, 8400, 84) veya (8400, 84)
2. **Split:** Her detection (84 değer) = 4 bbox + 80 class score
   - İlk 4: cx, cy, w, h (center format, 640x640 koordinatlarında)
   - Son 80: her COCO sınıfı için confidence score (sigmoid zaten uygulanmış)
3. **Confidence filtering:** Max class score < threshold (0.25) olanları at
4. **xywh → xyxy dönüşümü:**
   ```
   x1 = cx - w/2
   y1 = cy - h/2
   x2 = cx + w/2
   y2 = cy + h/2
   ```
5. **Scale to original image:** Letterbox padding'i hesaba kat

**Dosyalar:**
- `include/edge_ml/vision/detection.h`
- `src/vision/detection.cpp`

**Veri yapısı:**
```cpp
struct Detection {
    float x1, y1, x2, y2;    // bbox (orijinal görüntü koordinatları)
    float confidence;          // max class score
    int class_id;             // en yüksek score'a sahip sınıf
    std::string class_name;   // COCO sınıf adı
};
```

### Step 6.6 — NMS (Non-Maximum Suppression)

Aynı nesne için birden fazla detection var. NMS ile en iyi olanı seç:

**Algoritma:**
1. Tüm detection'ları confidence'a göre sırala (büyükten küçüğe)
2. En yüksek confidence'lı detection'ı al, sonuç listesine ekle
3. Bu detection ile kalan tüm detection'ların IoU'sunu hesapla
4. IoU >= threshold (0.45) olan detection'ları kaldır (aynı nesne)
5. Kalan detection'larla 2'den tekrarla
6. Liste bitene kadar devam et

**IoU (Intersection over Union) hesaplama:**
```
intersection_x1 = max(a.x1, b.x1)
intersection_y1 = max(a.y1, b.y1)
intersection_x2 = min(a.x2, b.x2)
intersection_y2 = min(a.y2, b.y2)

intersection_area = max(0, x2-x1) * max(0, y2-y1)
union_area = area(a) + area(b) - intersection_area

iou = intersection_area / union_area
```

**Fonksiyonlar:**
```cpp
float compute_iou(const Detection& a, const Detection& b);
std::vector<Detection> nms(std::vector<Detection>& detections, 
                           float iou_threshold = 0.45f);
std::vector<Detection> postprocess(const Tensor& model_output,
                                   float conf_threshold = 0.25f,
                                   float iou_threshold = 0.45f);
```

### Step 6.7 — Sonuç Görselleştirme

1. Detection'ları terminale yazdır (sınıf adı, confidence, bbox koordinatları)
2. Basit ASCII output veya PPM/BMP üzerine bbox çiz
3. COCO sınıf isimleri listesi (80 sınıf: person, bicycle, car, ...)

### Step 6.8 — Python Doğrulama

Engine sonuçlarını Python ONNXRuntime ile karşılaştır:

```python
# tools/validate_yolo.py
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("yolo11n.onnx")
output = session.run(None, {"images": input_tensor})

# C++ engine output ile karşılaştır
# Bbox koordinatları eşleşmeli (tolerance: 1 pixel)
# Class ID'ler eşleşmeli
# Confidence'lar yakın olmalı (tolerance: 0.01)
```

### Faz 6 Tamamlanma Kriterleri

- [ ] YOLO modeli başarıyla export ve parse edildi
- [ ] Image preprocessing pipeline çalışıyor
- [ ] End-to-end inference: image → model → raw output
- [ ] Bbox decode doğru çalışıyor
- [ ] NMS implement edildi ve doğru filtreleme yapıyor
- [ ] Python ONNXRuntime ile sonuçlar eşleşiyor (±1 pixel bbox, aynı class ID)
- [ ] En az 3 farklı test görüntüsü ile doğrulandı
- [ ] Branch: `phase-6-yolo` → PR → merge to main

---

## Faz 7 — Performans Optimizasyonu

Engine çalışıyor, şimdi hızlandırma zamanı.

### Step 7.1 — Profiling

1. Her operatörün çalışma süresini ölç (chrono)
2. Bottleneck analizi: hangi layer en yavaş?
3. Memory kullanımı analizi: peak memory ne kadar?

**Dosya:** `include/edge_ml/profiler.h`, `src/profiler.cpp`
```cpp
class Profiler {
    std::map<std::string, double> op_times;
    void start(const std::string& name);
    void stop(const std::string& name);
    void report();  // en yavaş → en hızlı sıralama
};
```

### Step 7.2 — Multi-Threaded Executor

Topological sort sırasında bağımsız node'ları paralel çalıştır:

1. Her topological level'daki node'lar birbirinden bağımsız
2. `std::thread` veya `std::async` ile paralelize et
3. Level-based execution: her level tamamlandığında sonraki level'a geç
4. Thread sayısı: `std::thread::hardware_concurrency()`

**Alternatif:** Operator-level paralelizm
- Büyük Conv operasyonlarını birden fazla thread'e böl
- Output channel'ları thread'lere dağıt

### Step 7.3 — Memory Pool

Sürekli malloc/free yerine önceden tahsis edilmiş bellek havuzu:

1. **Arena allocator:** Büyük bir blok tahsis et, içinden parçalar dağıt
2. **Tensor reuse:** Artık kullanılmayan tensor'ların belleğini yeni tensor'lara ver
3. **In-place operations:** Mümkün olduğunca yeni tensor oluşturmak yerine mevcut tensor'ı değiştir (ReLU, Sigmoid gibi)

```cpp
class MemoryPool {
    void* pool;
    size_t pool_size;
    size_t offset;
    
    void* allocate(size_t bytes);
    void reset();  // tüm allocation'ları sıfırla (inference arası)
};
```

### Step 7.4 — SIMD'i Tüm Operatörlere Yay

Şu an SIMD sadece add, multiply, relu, matmul'da var. Eklenecekler:
- `simd_sigmoid` — NEON ile hızlı sigmoid approximation
- `simd_conv2d` — im2col matmul'ü SIMD matmul ile değiştir
- `simd_concat` — Büyük memory copy'lerde SIMD kullan

### Step 7.5 — Quantized Inference (Opsiyonel)

Tüm model'i INT8'de çalıştır:
1. Tüm weight'leri quantize et
2. Her operatörün INT8 versiyonunu yaz
3. INT32 accumulation ile matmul
4. Activation'lar arasında requantize
5. Accuracy vs speed tradeoff ölç

### Step 7.6 — Benchmark Suite

```
./yolo_benchmark
  - Preprocessing time: X ms
  - Inference time: X ms
  - Postprocessing time: X ms
  - Total: X ms
  - FPS: X
  - Peak memory: X MB
  - Comparison: vs ONNXRuntime
```

### Faz 7 Tamamlanma Kriterleri

- [ ] Profiling her operatörün süresini gösteriyor
- [ ] En az 2x speedup (multi-thread veya SIMD genişletme ile)
- [ ] Memory kullanımı %30+ azaltıldı (pool/reuse ile)
- [ ] Benchmark suite tüm metrikleri raporluyor
- [ ] Branch: `phase-7-performance` → PR → merge to main

---

## Faz 8 — Demo & Portfolio

### Step 8.1 — Real-Time Demo

1. Webcam veya video dosyasından frame oku
2. Her frame'i preprocess → inference → postprocess
3. Sonuçları görselleştir (bbox + class label + confidence)
4. FPS counter göster

**Kütüphane seçenekleri (minimal dependency):**
- stb_image + stb_image_write (header-only, sıfır dependency)
- V4L2 (Linux webcam) veya AVFoundation (macOS webcam)
- Veya basit: video'dan frame'leri önceden extract et, sırayla işle

### Step 8.2 — Benchmark Karşılaştırma Sayfası

| Metric | Edge ML Engine | ONNXRuntime | Ratio |
|--------|---------------|-------------|-------|
| Inference time | X ms | Y ms | - |
| Memory usage | X MB | Y MB | - |
| Binary size | X MB | Y MB | - |
| Dependencies | 2 | 50+ | - |

### Step 8.3 — README Final Update

- Demo GIF/screenshot ekle
- Benchmark sonuçlarını güncelle
- Kurulum talimatları (one-liner build)
- API kullanım örnekleri

### Step 8.4 — Blog Post (Opsiyonel)

- "Building an ML Inference Engine from Scratch in C++"
- Teknik derinlik: SIMD, quantization, graph execution
- Lessons learned
- Medium veya kişisel blog

### Step 8.5 — Demo Video (Opsiyonel)

- 2-3 dakikalık video
- Proje mimarisi açıklama
- Canlı demo: YOLO object detection
- Terminal output + görselleştirme

### Faz 8 Tamamlanma Kriterleri

- [ ] Real-time veya batch demo çalışıyor
- [ ] Benchmark karşılaştırma tamamlandı
- [ ] README son haline getirildi
- [ ] (Opsiyonel) Blog post yayınlandı
- [ ] (Opsiyonel) Demo video hazırlandı
- [ ] Branch: `phase-8-demo` → PR → merge to main

---

## Zaman Tahmini

| Faz | Tahmini Süre | Zorluk |
|-----|-------------|--------|
| Faz 5 — Operatörler | 2-3 hafta | Orta |
| Faz 6 — YOLO Pipeline | 3-4 hafta | Yüksek |
| Faz 7 — Performans | 2-3 hafta | Orta-Yüksek |
| Faz 8 — Demo & Portfolio | 1-2 hafta | Düşük |
| **Toplam** | **8-12 hafta** | - |

---

## Bağımlılık Grafiği

```
Faz 5 (Operatörler)
  │
  ▼
Faz 6 (YOLO Pipeline)
  │
  ├──► Faz 7 (Performans)  ← Faz 6 tamamlanınca başlayabilir
  │
  └──► Faz 8 (Demo)        ← Faz 6 tamamlanınca başlayabilir
```

Faz 7 ve Faz 8 paralel ilerleyebilir. Faz 5 ve Faz 6 sıralı olmalı.
