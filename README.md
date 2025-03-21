# 📚 Hướng Dẫn Huấn Luyện Mô Hình FastSpeech2 Đa Ngôn Ngữ 🇻🇳 🇬🇧

## 📋 Giới Thiệu

FastSpeech2 là mô hình text-to-speech (TTS) tiên tiến, được phát triển bởi Microsoft, giải quyết nhiều vấn đề của các mô hình TTS truyền thống như chậm trong quá trình inference và thiếu khả năng kiểm soát giọng nói (như tốc độ, cao độ, năng lượng). Dự án này triển khai FastSpeech2 cho tiếng Việt và tiếng Anh, cho phép tạo giọng nói tự nhiên với khả năng kiểm soát các thuộc tính giọng nói.

## 🔄 Quy Trình Huấn Luyện

### 1️⃣ Thu thập dữ liệu

- **Tiếng Việt**: Sử dụng bộ dữ liệu [InfoRe](https://huggingface.co/datasets/ntt123/infore/resolve/main/infore_16k_denoised.zip) (hoặc tương tự)
- **Tiếng Anh**: Sử dụng bộ dữ liệu [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)

Mỗi bộ dữ liệu cần có:
- File âm thanh `.wav` (tốt nhất là 22.05kHz, 16-bit)
- File văn bản tương ứng (transcripts)

### 2️⃣ Căn chỉnh dữ liệu với Montreal Forced Aligner (MFA) 🔍

MFA được sử dụng để căn chỉnh âm thanh với văn bản ở cấp độ phoneme, tạo ra thông tin thời gian chính xác cho mỗi phoneme.

1. **Cài đặt MFA**:
```bash
pip install montreal-forced-aligner
```

2. **Chuẩn bị dữ liệu cho MFA**:
   - Tạo thư mục chứa các file âm thanh `.wav`
   - Tạo file `.lab` hoặc `.TextGrid` chứa nội dung văn bản tương ứng

3. **Thực hiện alignment**:
```bash
mfa align /đường_dẫn/đến/dữ_liệu /đường_dẫn/đến/từ_điển /đường_dẫn/đến/mô_hình_âm_vị tiếng_việt
```

4. **Kết quả alignment**:
   - File `.TextGrid` chứa thông tin thời gian cho từng phoneme
   - Dữ liệu này sẽ được sử dụng để huấn luyện mô hình duration predictor

### 3️⃣ Tiền xử lý và chuẩn hóa dữ liệu 🧹

```bash
python preprocess.py --config config/LJSpeech/preprocess.yaml
python preprocess.py --config config/infore/preprocess.yaml
```

Quá trình tiền xử lý bao gồm:

1. **Chuẩn hóa văn bản**:
   - Tiếng Anh: Chuyển đổi từ văn bản sang phoneme bằng cách sử dụng thư viện `g2p-en`
   - Tiếng Việt: Sử dụng `text.vietnamese_phonemes` để chuyển đổi thành phoneme tiếng Việt

2. **Trích xuất đặc trưng âm thanh**:
   - Xử lý tín hiệu âm thanh thành Mel spectrogram
   - Trích xuất thông tin pitch (F0) sử dụng PyWorld
   - Trích xuất thông tin energy từ mel spectrogram

3. **Chuẩn hóa**:
   - Chuẩn hóa độ dài dữ liệu
   - Tính toán thống kê (mean, std) của pitch và energy cho việc chuẩn hóa
   - Lưu trữ thông tin alignment để tính duration của mỗi phoneme

4. **Lưu trữ dữ liệu tiền xử lý**:
   - Dữ liệu được lưu trong thư mục `preprocessed_data`
   - Bao gồm mel spectrograms, thông tin pitch, energy, duration và text sequences

### 4️⃣ Huấn luyện mô hình 🚀

```bash
python train.py --config config/LJSpeech/preprocess.yaml config/LJSpeech/model.yaml config/LJSpeech/train.yaml
python train.py --config config/infore/preprocess.yaml config/infore/model.yaml config/infore/train.yaml
```

Quá trình huấn luyện:

1. **Kiến trúc FastSpeech2**:
   - **Encoder**: Biến đổi chuỗi phoneme thành biểu diễn hidden
   - **Variance Adaptor**: Dự đoán và điều chỉnh pitch, energy, duration
   - **Decoder**: Biến đổi biểu diễn hidden thành mel spectrogram
   - **Vocoder**: Biến đổi mel spectrogram thành dạng sóng âm thanh (HiFi-GAN)

2. **Các giai đoạn huấn luyện**:
   - Huấn luyện mô hình FastSpeech2 (encoder, variance adaptor, decoder)
   - Sử dụng vocoder được huấn luyện trước (HiFi-GAN) để chuyển đổi thành audio

3. **Chiến lược huấn luyện**:
   - Sử dụng Adam optimizer với scheduled learning rate
   - Huấn luyện với batch size 16-32 (tùy thuộc vào GPU)
   - Sử dụng gradient clipping để ổn định quá trình huấn luyện
   - Lưu checkpoint mô hình định kỳ để đánh giá

4. **Theo dõi quá trình huấn luyện**:
   - Sử dụng TensorBoard để theo dõi loss, mel spectrograms, và audio samples
   - Đánh giá mô hình qua các thời điểm checkpoint khác nhau

### 5️⃣ Đánh giá mô hình và tinh chỉnh 📊

```bash
python synthesize.py --restore_step 100000 --mode single --text "Xin chào, tôi là trợ lý ảo."
```

1. **Đánh giá chất lượng**:
   - Đánh giá chủ quan bằng cách nghe thử các mẫu âm thanh tạo ra
   - So sánh với ground truth và các mô hình TTS khác

2. **Tinh chỉnh**:
   - Điều chỉnh hyperparameters dựa trên kết quả đánh giá
   - Cân nhắc fine-tuning trên dữ liệu bổ sung nếu cần

### 6️⃣ Triển khai mô hình 🖥️

1. **Chuyển đổi mô hình**:
   - Sử dụng các checkpoint đã huấn luyện
   - Tích hợp với vocoder HiFi-GAN

2. **Tạo giao diện người dùng**:
   - Sử dụng PyQt5 để xây dựng giao diện đồ họa `tts_dual_mode.py`
   - Hỗ trợ đa ngôn ngữ với tính năng tự động phát hiện ngôn ngữ

3. **Cài đặt và sử dụng**:
   - Cài đặt các thư viện cần thiết: `pip install -r requirements.txt`
   - Chạy ứng dụng: `python tts_dual_mode.py`

## 🔧 Cấu trúc dự án

```
FastSpeech2_vi/
├── config/                 # Cấu hình cho từng ngôn ngữ
│   ├── LJSpeech/          # Cấu hình cho tiếng Anh
│   └── infore/            # Cấu hình cho tiếng Việt
├── dataset/               # Xử lý và tải dữ liệu
├── hifigan/               # Vocoder HiFi-GAN
├── model/                 # Mô hình FastSpeech2
│   ├── blocks.py          # Các khối building block
│   ├── variance_adaptor.py # Bộ điều chỉnh phương sai
│   └── ...
├── output/                # Kết quả và checkpoint
│   ├── ckpt/              # Checkpoint mô hình
│   └── result/            # Kết quả synthesis
├── preprocessed_data/     # Dữ liệu đã tiền xử lý
├── text/                  # Xử lý text và phoneme
├── utils/                 # Công cụ hỗ trợ
├── preprocess.py          # Script tiền xử lý
├── train.py               # Script huấn luyện
├── synthesize.py          # Tạo giọng nói từ mô hình
└── tts_dual_mode.py       # Ứng dụng GUI đa ngôn ngữ
```

## 🚀 Ưu điểm của FastSpeech2

1. **Tốc độ inference nhanh**: Kiến trúc non-autoregressive cho phép tạo ra âm thanh nhanh hơn nhiều lần so với các mô hình autoregressive như Tacotron 2.

2. **Kiểm soát linh hoạt**: Cho phép điều chỉnh pitch, energy và duration, tạo ra giọng nói với nhiều cảm xúc và nhấn mạnh khác nhau.

3. **Chất lượng cao**: Khả năng tạo ra giọng nói tự nhiên, rõ ràng với ít lỗi phổ biến của TTS (lặp từ, bỏ sót từ).

4. **Đa ngôn ngữ**: Dễ dàng mở rộng cho nhiều ngôn ngữ khác nhau, bao gồm cả tiếng Việt với hệ thống dấu thanh phức tạp.

## 📝 Tài liệu tham khảo

1. [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
2. [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)
3. [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)
4. [Text-to-Speech for Low-resource Languages: A Survey](https://arxiv.org/abs/2110.04040)

## 🎯 Hướng dẫn chi tiết train mô hình FastSpeech2

Để tham khảo cách train mô hình FastSpeech2 đầy đủ hơn, bạn có thể tham khảo repo của tác giả:

🔗 [FastSpeech2](https://github.com/ming024/FastSpeech2)


---

📱 **Tác giả**: Hoài Nhân  
🌐 **Liên hệ**: hoainhannro@gmail.com  
📅 **Cập nhật**: Tháng 3, 2025
