# 📚 Hướng Dẫn Huấn Luyện & Triển Khai FastSpeech2 Đa Ngôn Ngữ 🇻🇳 🇬🇧

## 📋 Giới Thiệu

FastSpeech2 là mô hình text-to-speech (TTS) tiên tiến, được phát triển bởi Microsoft, giải quyết nhiều vấn đề của các mô hình TTS truyền thống như chậm trong quá trình inference và thiếu khả năng kiểm soát giọng nói (như tốc độ, cao độ, năng lượng). Dự án này triển khai FastSpeech2 cho tiếng Việt và tiếng Anh, cho phép tạo giọng nói tự nhiên với khả năng kiểm soát các thuộc tính giọng nói.

## 🚀 Hướng Dẫn Cài Đặt Nhanh

```bash
# Clone repo
https://github.com/hoainhanpro/En_Vi_TextToSpeech.git
cd En_Vi_TextToSpeech

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

## 🔄 Quy Trình Huấn Luyện

### 1️⃣ Thu thập dữ liệu
- **Tiếng Việt**: [InfoRe](https://huggingface.co/datasets/ntt123/infore/resolve/main/infore_16k_denoised.zip)
- **Tiếng Anh**: [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)
- Mỗi bộ dữ liệu cần có file `.wav` (22.05kHz, 16-bit) và transcript `.txt`

### 2️⃣ Căn chỉnh dữ liệu với Montreal Forced Aligner (MFA)
```bash
pip install montreal-forced-aligner
# Thực hiện alignment
mfa align /duong_dan/den/du_lieu /duong_dan/den/tu_dien /duong_dan/den/mo_hinh_am_vi tieng_viet
```
- Kết quả: file `.TextGrid` cho mỗi audio, dùng để train duration predictor

### 3️⃣ Tiền xử lý và chuẩn hóa dữ liệu
```bash
python preprocess.py --config config/LJSpeech/preprocess.yaml
python preprocess.py --config config/infore/preprocess.yaml
```
- Chuẩn hóa văn bản (g2p-en cho tiếng Anh, text.vietnamese_phonemes cho tiếng Việt)
- Trích xuất Mel spectrogram, pitch (PyWorld), energy
- Lưu dữ liệu vào `preprocessed_data/`

### 4️⃣ Huấn luyện mô hình
```bash
python train.py --config config/LJSpeech/preprocess.yaml config/LJSpeech/model.yaml config/LJSpeech/train.yaml
python train.py --config config/infore/preprocess.yaml config/infore/model.yaml config/infore/train.yaml
```
- Theo dõi loss, mel spectrogram, audio bằng TensorBoard
- Checkpoint lưu ở `output/ckpt/`

### 5️⃣ Đánh giá mô hình và tinh chỉnh
```bash
python synthesize.py --restore_step 100000 --mode single --text "Xin chào, tôi là trợ lý ảo."
```
- Đánh giá chủ quan bằng nghe thử, so sánh với ground truth
- Có thể tinh chỉnh hyperparameters hoặc fine-tune thêm

### 6️⃣ Triển khai mô hình & Web Demo
- Tích hợp với vocoder HiFi-GAN
- Có thể chạy giao diện web demo:
```bash
python run_web_server.py
# hoặc
python tts_web_app.py
```
- Truy cập: http://localhost:5000 để sử dụng giao diện web (điều chỉnh pitch, energy, duration, chọn chế độ đa ngôn ngữ)
- Có thể sử dụng notebook hướng dẫn: `vn-text-to-speech.ipynb`

## 🔧 Cấu trúc dự án

```
En_Vi_TextToSpeech/
├── audio/                 # Xử lý tín hiệu âm thanh
├── config/                # Cấu hình cho từng ngôn ngữ
├── hifigan/               # Vocoder HiFi-GAN
├── lexicon/               # Từ điển ngữ âm
├── mfa/                   # Montreal Forced Aligner
├── model/                 # Mô hình FastSpeech2
├── output/                # Checkpoint, kết quả
├── preprocessed_data/     # Dữ liệu đã tiền xử lý
├── preprocessor/          # Scripts tiền xử lý
├── scripts/               # Scripts hỗ trợ
├── static/, templates/    # Giao diện web
├── text/                  # Xử lý text và phoneme
├── transformer/           # Mô hình transformer
├── utils/                 # Công cụ hỗ trợ
├── dataset.py, train.py, synthesize.py, preprocess.py, evaluate.py
├── run_web_server.py, tts_web_app.py # Web server/demo
├── vn-text-to-speech.ipynb # Notebook hướng dẫn
└── README.md
```

## 💡 Tính năng nổi bật
- Hỗ trợ tiếng Việt & tiếng Anh, dễ mở rộng thêm ngôn ngữ
- Điều chỉnh linh hoạt pitch, energy, duration từ giao diện web
- Chế độ đa ngôn ngữ: tự động phát hiện và tách đoạn văn bản theo ngôn ngữ
- Tích hợp vocoder HiFi-GAN cho chất lượng âm thanh cao

## 📝 Tài liệu tham khảo
1. [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
2. [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)
3. [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)
4. [Text-to-Speech for Low-resource Languages: A Survey](https://arxiv.org/abs/2110.04040)
## Để tham khảo cách train mô hình FastSpeech2 đầy đủ hơn, bạn có thể tham khảo repo của tác giả:

🔗 [FastSpeech2](https://github.com/ming024/FastSpeech2)
---


📱 **Tác giả**: Hoài Nhân  
🌐 **Liên hệ**: hoainhannro@gmail.com  
📅 **Cập nhật**: Tháng 3, 2025
