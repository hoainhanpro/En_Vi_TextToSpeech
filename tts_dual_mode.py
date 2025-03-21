import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QLabel,
                             QSlider, QFileDialog, QMessageBox, QTabWidget,
                             QComboBox, QRadioButton, QButtonGroup, QCheckBox)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import argparse
import torch
import re
from string import punctuation
from langdetect import detect, detect_langs, LangDetectException
import tempfile
from pydub import AudioSegment

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples, plot_mel
from dataset import TextDataset
from text import text_to_sequence
import text.vietnamese_phonemes as viphonemes

# Import các hàm từ synthesize.py
from synthesize import (read_lexicon, preprocess_vietnamese, 
                      preprocess_english, clean_vietnamese_text)

class SpectrogramCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super(SpectrogramCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
        
    def plot_spectrogram(self, mel, pitch, energy, stats):
        self.ax.clear()
        pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
        pitch = pitch * pitch_std + pitch_mean
        
        self.ax.imshow(mel, origin="lower", aspect="auto")
        self.ax.set_title("Spectrogram", fontsize="medium")
        self.ax.tick_params(labelsize="x-small", left=False, labelleft=False)
        
        ax1 = self.ax.twinx()
        ax1.plot(pitch, color="tomato")
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(labelsize="x-small", colors="tomato")
        
        ax2 = self.ax.twinx()
        ax2.spines["right"].set_position(("axes", 1.1))
        ax2.plot(energy, color="darkviolet")
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.tick_params(labelsize="x-small", colors="darkviolet")
        
        self.fig.tight_layout()
        self.draw()

class DualModeTTSGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Cấu hình mặc định
        self.vi_preprocess_config_path = "config/infore/preprocess.yaml"
        self.vi_model_config_path = "config/infore/model.yaml"
        self.vi_train_config_path = "config/infore/train.yaml"
        self.vi_restore_step = 100000
        
        self.en_preprocess_config_path = "config/LJSpeech/preprocess.yaml"
        self.en_model_config_path = "config/LJSpeech/model.yaml"
        self.en_train_config_path = "config/LJSpeech/train.yaml"
        self.en_restore_step = 900000
        
        # Các thông số điều chỉnh
        self.pitch_control = 1.0
        self.energy_control = 1.0
        self.duration_control = 1.0
        
        # Đường dẫn lưu file âm thanh
        self.audio_path = None
        
        # Media player
        self.player = QMediaPlayer()
        
        # Ngôn ngữ hiện tại
        self.current_language = "vi"  # Mặc định là tiếng Việt
        
        # Chế độ đa ngôn ngữ
        self.multi_language_mode = False
        
        # Models
        self.vi_model = None
        self.vi_vocoder = None
        self.vi_preprocess_config = None
        self.vi_model_config = None
        self.vi_train_config = None
        self.vi_stats = None
        
        self.en_model = None
        self.en_vocoder = None
        self.en_preprocess_config = None
        self.en_model_config = None
        self.en_train_config = None
        self.en_stats = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.init_ui()
        
    def detect_language(self, text):
        """Tự động nhận dạng ngôn ngữ của văn bản"""
        try:
            # Chia văn bản thành các câu
            sentences = text.split('.')
            vi_count = 0
            en_count = 0
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                try:
                    # Sử dụng langdetect để nhận dạng ngôn ngữ
                    lang = detect(sentence)
                    
                    # Đếm số lượng câu theo ngôn ngữ
                    if lang == 'vi':
                        vi_count += 1
                    elif lang == 'en':
                        en_count += 1
                except LangDetectException:
                    continue
            
            # So sánh số lượng câu để quyết định ngôn ngữ chính
            if vi_count >= en_count:
                return "vi"
            else:
                return "en"
                
        except Exception as e:
            print(f"Lỗi nhận dạng ngôn ngữ: {e}")
            return "vi"  # Mặc định là tiếng Việt nếu có lỗi
            
    def detect_language_for_segment(self, text):
        """Nhận dạng ngôn ngữ cho một đoạn văn bản ngắn"""
        try:
            lang = detect(text)
            if lang == 'vi':
                return "vi"
            elif lang == 'en':
                return "en"
            return "vi"  # Mặc định là tiếng Việt
        except Exception as e:
            print(f"Lỗi nhận dạng ngôn ngữ cho đoạn văn bản: {e}")
            return "vi"  # Mặc định là tiếng Việt nếu có lỗi
    
    def split_text_by_language(self, text):
        """Tách văn bản thành các đoạn theo ngôn ngữ"""
        # Tách văn bản thành các từ và dấu câu
        segments = []
        current_segment = ""
        current_lang = None
        
        # Tách văn bản thành các từ và cụm từ
        words = re.findall(r'\w+|[^\w\s]', text)
        
        for word in words:
            if not word.strip():
                continue
                
            # Nếu từ chỉ chứa dấu câu, thêm vào đoạn hiện tại
            if not re.search(r'\w', word):
                if current_segment:
                    current_segment += word
                continue
                
            # Thử nhận dạng ngôn ngữ cho cụm từ đủ dài
            if len(word) > 3:
                try:
                    word_lang = self.detect_language_for_segment(word)
                    
                    # Nếu đoạn hiện tại trống hoặc cùng ngôn ngữ, thêm từ vào
                    if current_lang is None:
                        current_lang = word_lang
                        current_segment = word
                    elif current_lang == word_lang:
                        current_segment += " " + word
                    else:
                        # Nếu khác ngôn ngữ, lưu đoạn hiện tại và bắt đầu đoạn mới
                        if current_segment:
                            segments.append((current_segment, current_lang))
                        current_segment = word
                        current_lang = word_lang
                except:
                    # Nếu không nhận dạng được, thêm vào đoạn hiện tại
                    if current_segment:
                        current_segment += " " + word
                    else:
                        current_segment = word
                        current_lang = "vi"  # Mặc định
            else:
                # Từ quá ngắn, thêm vào đoạn hiện tại
                if current_segment:
                    current_segment += " " + word
                else:
                    current_segment = word
                    current_lang = "vi"  # Mặc định
        
        # Thêm đoạn cuối cùng nếu có
        if current_segment:
            segments.append((current_segment, current_lang))
            
        return segments

    def init_ui(self):
        self.setWindowTitle('Đa Ngôn Ngữ Text-to-Speech 🌍')
        self.setGeometry(100, 100, 1000, 800)
        
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Text input section
        self.text_label = QLabel("Nhập văn bản:")
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Nhập văn bản vào đây...")
        self.text_input.textChanged.connect(self.on_text_changed)
        
        # Language detection label
        self.language_label = QLabel("Ngôn ngữ phát hiện: Chưa có văn bản")
        self.language_label.setStyleSheet("color: blue;")
        
        # Multilanguage mode checkbox
        self.multi_language_checkbox = QCheckBox("Chế độ đa ngôn ngữ (xử lý từng phần văn bản theo ngôn ngữ riêng) 🔄")
        self.multi_language_checkbox.toggled.connect(self.toggle_multi_language_mode)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.generate_button = QPushButton("Tạo Audio 🔊")
        self.generate_button.clicked.connect(self.generate_speech)
        self.play_button = QPushButton("Phát Audio ▶️")
        self.play_button.clicked.connect(self.play_audio)
        self.save_button = QPushButton("Lưu Audio 💾")
        self.save_button.clicked.connect(self.save_audio)
        self.load_models_button = QPushButton("Tải Models 📥")
        self.load_models_button.clicked.connect(self.load_both_models)
        
        button_layout.addWidget(self.generate_button)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.load_models_button)
        
        # Control sliders
        control_layout = QVBoxLayout()
        
        pitch_layout = QHBoxLayout()
        pitch_label = QLabel("Độ cao:")
        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setRange(50, 150)
        self.pitch_slider.setValue(100)
        self.pitch_slider.valueChanged.connect(self.update_pitch)
        self.pitch_value_label = QLabel("1.0")
        pitch_layout.addWidget(pitch_label)
        pitch_layout.addWidget(self.pitch_slider)
        pitch_layout.addWidget(self.pitch_value_label)
        
        energy_layout = QHBoxLayout()
        energy_label = QLabel("Âm lượng:")
        self.energy_slider = QSlider(Qt.Horizontal)
        self.energy_slider.setRange(50, 150)
        self.energy_slider.setValue(100)
        self.energy_slider.valueChanged.connect(self.update_energy)
        self.energy_value_label = QLabel("1.0")
        energy_layout.addWidget(energy_label)
        energy_layout.addWidget(self.energy_slider)
        energy_layout.addWidget(self.energy_value_label)
        
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Tốc độ:")
        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.setRange(50, 150)
        self.duration_slider.setValue(100)
        self.duration_slider.valueChanged.connect(self.update_duration)
        self.duration_value_label = QLabel("1.0")
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_slider)
        duration_layout.addWidget(self.duration_value_label)
        
        control_layout.addLayout(pitch_layout)
        control_layout.addLayout(energy_layout)
        control_layout.addLayout(duration_layout)
        
        # Spectrogram display
        self.canvas = SpectrogramCanvas(self, width=8, height=3)
        
        # Status section
        self.status_label = QLabel("Trạng thái: Vui lòng tải models...")
        self.status_label.setStyleSheet("color: blue;")
        
        # Assemble the layout
        main_layout.addWidget(self.text_label)
        main_layout.addWidget(self.text_input)
        main_layout.addWidget(self.language_label)
        main_layout.addWidget(self.multi_language_checkbox)
        main_layout.addLayout(button_layout)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.status_label)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def toggle_multi_language_mode(self, checked):
        """Chuyển đổi chế độ đa ngôn ngữ"""
        self.multi_language_mode = checked
        if checked:
            self.status_label.setText("Trạng thái: Đã bật chế độ đa ngôn ngữ ✅")
            self.status_label.setStyleSheet("color: green;")
        else:
            text = self.text_input.toPlainText().strip()
            if text:
                detected_lang = self.detect_language(text)
                self.current_language = detected_lang
                lang_name = "Tiếng Việt 🇻🇳" if detected_lang == "vi" else "Tiếng Anh 🇬🇧"
                self.language_label.setText(f"Ngôn ngữ phát hiện: {lang_name}")
            self.status_label.setText("Trạng thái: Đã tắt chế độ đa ngôn ngữ")
            self.status_label.setStyleSheet("color: blue;")
        
    def on_text_changed(self):
        """Xử lý khi văn bản thay đổi"""
        text = self.text_input.toPlainText().strip()
        if text:
            if self.multi_language_mode:
                segments = self.split_text_by_language(text)
                lang_info = []
                for segment, lang in segments:
                    lang_name = "Tiếng Việt" if lang == "vi" else "Tiếng Anh"
                    lang_info.append(f"{segment} ({lang_name})")
                
                if lang_info:
                    self.language_label.setText(f"Phân đoạn ngôn ngữ: {len(segments)} đoạn văn bản")
                    self.status_label.setText(f"Đã phân tích thành {len(segments)} đoạn văn bản khác nhau")
                    self.status_label.setStyleSheet("color: green;")
                else:
                    self.language_label.setText("Ngôn ngữ phát hiện: Chưa thể phân đoạn")
                    self.language_label.setStyleSheet("color: orange;")
            else:
                detected_lang = self.detect_language(text)
                self.current_language = detected_lang
                lang_name = "Tiếng Việt 🇻🇳" if detected_lang == "vi" else "Tiếng Anh 🇬🇧"
                self.language_label.setText(f"Ngôn ngữ phát hiện: {lang_name}")
                self.language_label.setStyleSheet("color: green;")
        else:
            self.language_label.setText("Ngôn ngữ phát hiện: Chưa có văn bản")
            self.language_label.setStyleSheet("color: blue;")

    def generate_speech(self):
        try:
            text = self.text_input.toPlainText().strip()
            if not text:
                QMessageBox.warning(self, "Cảnh báo ⚠️", "Vui lòng nhập văn bản!")
                return
            
            # Kiểm tra model đã tải chưa
            if not self.vi_model and not self.en_model:
                QMessageBox.warning(self, "Cảnh báo ⚠️", "Vui lòng tải ít nhất một model trước!")
                return
            elif not self.vi_model and (self.current_language == "vi" or self.multi_language_mode):
                QMessageBox.warning(self, "Cảnh báo ⚠️", "Vui lòng tải model tiếng Việt trước!")
                return
            elif not self.en_model and (self.current_language == "en" or self.multi_language_mode):
                QMessageBox.warning(self, "Cảnh báo ⚠️", "Vui lòng tải model tiếng Anh trước!")
                return
            
            self.status_label.setText("Trạng thái: Đang tạo audio... 🔄")
            self.status_label.setStyleSheet("color: blue;")
            QApplication.processEvents()
            
            if self.multi_language_mode:
                self.generate_multi_language_speech(text)
            else:
                self.generate_single_language_speech(text)
                
        except Exception as e:
            self.status_label.setText(f"Trạng thái: Lỗi tạo audio! ❌")
            self.status_label.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Lỗi ❌", f"Không thể tạo audio: {str(e)}")
            print(f"Error generating speech: {e}")
    
    def generate_single_language_speech(self, text):
        """Tạo giọng nói sử dụng một ngôn ngữ"""
        # Chọn config và model theo ngôn ngữ
        if self.current_language == "vi":
            model = self.vi_model
            vocoder = self.vi_vocoder
            preprocess_config = self.vi_preprocess_config
            model_config = self.vi_model_config
            train_config = self.vi_train_config
            stats = self.vi_stats
            text_sequence = preprocess_vietnamese(text, preprocess_config)
            output_dir = "output/result/temp_vi"
        else:  # English
            model = self.en_model
            vocoder = self.en_vocoder
            preprocess_config = self.en_preprocess_config
            model_config = self.en_model_config
            train_config = self.en_train_config
            stats = self.en_stats
            text_sequence = preprocess_english(text, preprocess_config)
            output_dir = "output/result/temp_en"
        
        # Tạo batch giống như trong synthesize.py
        ids = raw_texts = [text[:100]]
        speakers = np.array([0])  # Speaker ID
        texts = np.array([text_sequence])
        text_lens = np.array([len(text_sequence)])
        batch = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
        
        # Tổng hợp giọng nói
        control_values = (self.pitch_control, self.energy_control, self.duration_control)
        
        with torch.no_grad():
            # Convert batch to device
            batch_device = to_device(batch[0], self.device)
            
            # Chạy model
            output = model(
                *(batch_device[2:]),
                p_control=control_values[0],
                e_control=control_values[1],
                d_control=control_values[2]
            )
            
            # Extract data từ output
            src_len = output[8][0].item()
            mel_len = output[9][0].item()
            mel_prediction = output[1][0, :mel_len].detach().transpose(0, 1)
            duration = output[5][0, :src_len].detach().cpu().numpy()
            
            # Xử lý pitch
            if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
                pitch = output[2][0, :src_len].detach().cpu().numpy()
                from utils.tools import expand
                pitch = expand(pitch, duration)
            else:
                pitch = output[2][0, :mel_len].detach().cpu().numpy()
            
            # Xử lý energy
            if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
                energy = output[3][0, :src_len].detach().cpu().numpy()
                from utils.tools import expand
                energy = expand(energy, duration)
            else:
                energy = output[3][0, :mel_len].detach().cpu().numpy()
            
            # Hiển thị spectrogram
            self.canvas.plot_spectrogram(mel_prediction.cpu().numpy(), pitch, energy, stats)
            
            # Tạo file âm thanh
            from utils.model import vocoder_infer
            
            mel_predictions = output[1].transpose(1, 2)
            lengths = output[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
            wav_predictions = vocoder_infer(
                mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
            )
            
            # Lưu file tạm để phát
            sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
            import scipy.io.wavfile as wavfile
            
            # Tạo thư mục output nếu chưa tồn tại
            os.makedirs(output_dir, exist_ok=True)
            
            self.audio_path = f"{output_dir}/output.wav"
            wavfile.write(self.audio_path, sampling_rate, wav_predictions[0])
            
            self.status_label.setText(f"Trạng thái: Đã tạo audio {self.current_language.upper()} thành công! ✅")
            self.status_label.setStyleSheet("color: green;")
            QMessageBox.information(self, "Thành công ✅", f"Đã tạo audio thành công ({self.current_language.upper()})!")
    
    def generate_multi_language_speech(self, text):
        """Tạo giọng nói đa ngôn ngữ"""
        # Phân tách văn bản thành các đoạn theo ngôn ngữ
        segments = self.split_text_by_language(text)
        
        if not segments:
            QMessageBox.warning(self, "Cảnh báo ⚠️", "Không thể phân tách văn bản thành các đoạn ngôn ngữ!")
            return
            
        self.status_label.setText(f"Trạng thái: Đang tạo audio cho {len(segments)} đoạn văn bản... 🔄")
        QApplication.processEvents()
        
        # Tạo thư mục tạm nếu chưa tồn tại
        output_dir = "output/result/temp_multi"
        os.makedirs(output_dir, exist_ok=True)
        
        # Tạo danh sách các file âm thanh tạm
        temp_audio_files = []
        
        # Tạo âm thanh cho từng đoạn
        for i, (segment_text, lang) in enumerate(segments):
            try:
                # Chọn config và model theo ngôn ngữ của đoạn
                if lang == "vi":
                    if not self.vi_model:
                        continue  # Bỏ qua nếu không có model tiếng Việt
                        
                    model = self.vi_model
                    vocoder = self.vi_vocoder
                    preprocess_config = self.vi_preprocess_config
                    model_config = self.vi_model_config
                    train_config = self.vi_train_config
                    text_sequence = preprocess_vietnamese(segment_text, preprocess_config)
                else:  # English
                    if not self.en_model:
                        continue  # Bỏ qua nếu không có model tiếng Anh
                        
                    model = self.en_model
                    vocoder = self.en_vocoder
                    preprocess_config = self.en_preprocess_config
                    model_config = self.en_model_config
                    train_config = self.en_train_config
                    text_sequence = preprocess_english(segment_text, preprocess_config)
                
                # Tạo batch
                ids = raw_texts = [segment_text[:100]]
                speakers = np.array([0])  # Speaker ID
                texts = np.array([text_sequence])
                text_lens = np.array([len(text_sequence)])
                batch = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
                
                # Tổng hợp giọng nói
                control_values = (self.pitch_control, self.energy_control, self.duration_control)
                
                with torch.no_grad():
                    # Convert batch to device
                    batch_device = to_device(batch[0], self.device)
                    
                    # Chạy model
                    output = model(
                        *(batch_device[2:]),
                        p_control=control_values[0],
                        e_control=control_values[1],
                        d_control=control_values[2]
                    )
                    
                    # Tạo file âm thanh
                    from utils.model import vocoder_infer
                    
                    mel_predictions = output[1].transpose(1, 2)
                    lengths = output[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
                    wav_predictions = vocoder_infer(
                        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
                    )
                    
                    # Lưu file tạm
                    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
                    import scipy.io.wavfile as wavfile
                    
                    temp_file = f"{output_dir}/segment_{i}.wav"
                    wavfile.write(temp_file, sampling_rate, wav_predictions[0])
                    temp_audio_files.append((temp_file, sampling_rate))
                    
                    self.status_label.setText(f"Trạng thái: Đã tạo audio cho đoạn {i+1}/{len(segments)}... 🔄")
                    QApplication.processEvents()
                    
            except Exception as e:
                print(f"Lỗi tạo audio cho đoạn {i}: {e}")
                continue
        
        if not temp_audio_files:
            QMessageBox.warning(self, "Cảnh báo ⚠️", "Không thể tạo audio cho bất kỳ đoạn nào!")
            return
            
        # Ghép các file âm thanh
        try:
            combined_audio = None
            
            for temp_file, _ in temp_audio_files:
                segment_audio = AudioSegment.from_wav(temp_file)
                
                if combined_audio is None:
                    combined_audio = segment_audio
                else:
                    combined_audio += segment_audio
            
            # Lưu file kết quả
            self.audio_path = f"{output_dir}/output_combined.wav"
            combined_audio.export(self.audio_path, format="wav")
            
            # Hiển thị thông báo thành công
            self.status_label.setText(f"Trạng thái: Đã tạo audio đa ngôn ngữ thành công! ✅")
            self.status_label.setStyleSheet("color: green;")
            QMessageBox.information(self, "Thành công ✅", f"Đã tạo audio đa ngôn ngữ thành công!")
            
        except Exception as e:
            print(f"Lỗi ghép file âm thanh: {e}")
            QMessageBox.warning(self, "Cảnh báo ⚠️", f"Lỗi khi ghép file âm thanh: {e}")
            
            # Nếu không ghép được, sử dụng file cuối cùng làm kết quả
            if temp_audio_files:
                self.audio_path = temp_audio_files[-1][0]
                self.status_label.setText(f"Trạng thái: Không thể ghép file âm thanh, hiển thị đoạn cuối cùng! ⚠️")
                self.status_label.setStyleSheet("color: orange;")

    def load_both_models(self):
        self.status_label.setText("Trạng thái: Đang tải models... 🔄")
        self.status_label.setStyleSheet("color: blue;")
        QApplication.processEvents()
        
        # Tải model tiếng Việt
        try:
            self.load_vietnamese_model()
            vi_loaded = True
        except Exception as e:
            vi_loaded = False
            print(f"Error loading Vietnamese model: {e}")
        
        # Tải model tiếng Anh
        try:
            self.load_english_model()
            en_loaded = True
        except Exception as e:
            en_loaded = False
            print(f"Error loading English model: {e}")
        
        # Hiển thị kết quả
        if vi_loaded and en_loaded:
            self.status_label.setText("Trạng thái: Đã tải cả hai models thành công! ✅")
            self.status_label.setStyleSheet("color: green;")
            QMessageBox.information(self, "Thành công ✅", "Đã tải cả hai models thành công!")
        elif vi_loaded:
            self.status_label.setText("Trạng thái: Đã tải model tiếng Việt, model tiếng Anh thất bại! ⚠️")
            self.status_label.setStyleSheet("color: orange;")
            QMessageBox.warning(self, "Cảnh báo ⚠️", "Chỉ tải được model tiếng Việt, model tiếng Anh thất bại!")
        elif en_loaded:
            self.status_label.setText("Trạng thái: Đã tải model tiếng Anh, model tiếng Việt thất bại! ⚠️")
            self.status_label.setStyleSheet("color: orange;")
            QMessageBox.warning(self, "Cảnh báo ⚠️", "Chỉ tải được model tiếng Anh, model tiếng Việt thất bại!")
        else:
            self.status_label.setText("Trạng thái: Không thể tải cả hai models! ❌")
            self.status_label.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Lỗi ❌", "Không thể tải cả hai models!")
        
    def load_vietnamese_model(self):
        # Đọc config
        self.vi_preprocess_config = yaml.load(
            open(self.vi_preprocess_config_path, "r"), Loader=yaml.FullLoader
        )
        self.vi_model_config = yaml.load(
            open(self.vi_model_config_path, "r"), Loader=yaml.FullLoader
        )
        self.vi_train_config = yaml.load(
            open(self.vi_train_config_path, "r"), Loader=yaml.FullLoader
        )
        
        # Tạo pseudo args để tương thích với hàm get_model
        class Args:
            def __init__(self, restore_step):
                self.restore_step = restore_step
        
        args = Args(self.vi_restore_step)
        
        # Tải model
        self.vi_model = get_model(
            args,
            (self.vi_preprocess_config, self.vi_model_config, self.vi_train_config),
            self.device,
            train=False
        )
        
        # Tải vocoder
        self.vi_vocoder = get_vocoder(self.vi_model_config, self.device)
        
        # Đọc stats cho việc hiển thị spectrogram
        with open(
            os.path.join(self.vi_preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            import json
            stats = json.load(f)
            self.vi_stats = stats["pitch"] + stats["energy"][:2]
    
    def load_english_model(self):
        # Đọc config
        self.en_preprocess_config = yaml.load(
            open(self.en_preprocess_config_path, "r"), Loader=yaml.FullLoader
        )
        self.en_model_config = yaml.load(
            open(self.en_model_config_path, "r"), Loader=yaml.FullLoader
        )
        self.en_train_config = yaml.load(
            open(self.en_train_config_path, "r"), Loader=yaml.FullLoader
        )
        
        # Tạo pseudo args để tương thích với hàm get_model
        class Args:
            def __init__(self, restore_step):
                self.restore_step = restore_step
        
        args = Args(self.en_restore_step)
        
        # Tải model
        self.en_model = get_model(
            args,
            (self.en_preprocess_config, self.en_model_config, self.en_train_config),
            self.device,
            train=False
        )
        
        # Tải vocoder
        self.en_vocoder = get_vocoder(self.en_model_config, self.device)
        
        # Đọc stats cho việc hiển thị spectrogram
        with open(
            os.path.join(self.en_preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            import json
            stats = json.load(f)
            self.en_stats = stats["pitch"] + stats["energy"][:2]
    
    def update_pitch(self):
        self.pitch_control = self.pitch_slider.value() / 100
        self.pitch_value_label.setText(f"{self.pitch_control:.1f}")
        
    def update_energy(self):
        self.energy_control = self.energy_slider.value() / 100
        self.energy_value_label.setText(f"{self.energy_control:.1f}")
        
    def update_duration(self):
        self.duration_control = self.duration_slider.value() / 100
        self.duration_value_label.setText(f"{self.duration_control:.1f}")
    
    def play_audio(self):
        if not self.audio_path or not os.path.exists(self.audio_path):
            QMessageBox.warning(self, "Cảnh báo ⚠️", "Chưa có file audio nào được tạo!")
            return
            
        try:
            self.status_label.setText("Trạng thái: Đang phát audio... 🔊")
            self.status_label.setStyleSheet("color: blue;")
            
            # Phát file âm thanh
            url = QUrl.fromLocalFile(os.path.abspath(self.audio_path))
            content = QMediaContent(url)
            self.player.setMedia(content)
            self.player.play()
            
            # Kết nối sự kiện kết thúc phát
            self.player.stateChanged.connect(self.on_player_state_changed)
            
        except Exception as e:
            self.status_label.setText(f"Trạng thái: Lỗi phát audio! ❌")
            self.status_label.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Lỗi ❌", f"Không thể phát audio: {str(e)}")
    
    def on_player_state_changed(self, state):
        if state == QMediaPlayer.StoppedState:
            self.status_label.setText("Trạng thái: Sẵn sàng ✅")
            self.status_label.setStyleSheet("color: green;")
    
    def save_audio(self):
        if not self.audio_path or not os.path.exists(self.audio_path):
            QMessageBox.warning(self, "Cảnh báo ⚠️", "Chưa có file audio nào được tạo!")
            return
            
        try:
            # Mở hộp thoại lưu file
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Lưu File Audio", "", "WAV Files (*.wav);;All Files (*)"
            )
            
            if file_path:
                # Sao chép file
                import shutil
                shutil.copy2(self.audio_path, file_path)
                self.status_label.setText(f"Trạng thái: Đã lưu file audio thành công! ✅")
                self.status_label.setStyleSheet("color: green;")
                QMessageBox.information(self, "Thành công ✅", f"Đã lưu file audio tại: {file_path}")
        except Exception as e:
            self.status_label.setText(f"Trạng thái: Lỗi lưu audio! ❌")
            self.status_label.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Lỗi ❌", f"Không thể lưu audio: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DualModeTTSGUI()
    window.show()
    sys.exit(app.exec_()) 