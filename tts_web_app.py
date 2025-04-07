import os
import sys
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend non-interactive
import matplotlib.pyplot as plt
import io
import base64
import json
import torch
import re
from string import punctuation
from langdetect import detect, detect_langs, LangDetectException
from pydub import AudioSegment
import uuid
from flask import Flask, request, jsonify, render_template, send_file, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import các hàm từ các module khác
from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
# Import các hàm từ synthesize.py
from synthesize import (read_lexicon, preprocess_vietnamese, 
                       preprocess_english, clean_vietnamese_text)

app = Flask(__name__, static_folder='static')
CORS(app)  # Cho phép cross-origin requests

# Cấu hình
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output/result/web_output'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # Giới hạn kích thước file upload (32MB)
app.config['ALLOWED_EXTENSIONS'] = {'pth.tar', 'pth', 'tar'}

# Tạo thư mục nếu chưa tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Cấu hình mặc định
CONFIG = {
    'vi': {
        'preprocess_config_path': "config/infore/preprocess.yaml",
        'model_config_path': "config/infore/model.yaml",
        'train_config_path': "config/infore/train.yaml",
        'restore_step': 100000
    },
    'en': {
        'preprocess_config_path': "config/LJSpeech/preprocess.yaml",
        'model_config_path': "config/LJSpeech/model.yaml",
        'train_config_path': "config/LJSpeech/train.yaml",
        'restore_step': 900000
    }
}

# Biến toàn cục để lưu trữ models đã tải
models = {
    'vi': {
        'model': None,
        'vocoder': None,
        'preprocess_config': None,
        'model_config': None,
        'train_config': None,
        'stats': None
    },
    'en': {
        'model': None,
        'vocoder': None,
        'preprocess_config': None,
        'model_config': None,
        'train_config': None,
        'stats': None
    }
}

def detect_language(text):
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

def detect_language_for_segment(text):
    """Nhận dạng ngôn ngữ cho một đoạn văn bản ngắn"""
    try:
        # Loại bỏ khoảng trắng thừa và dấu câu
        cleaned_text = text.strip()
        if not cleaned_text:
            return "vi"  # Mặc định là tiếng Việt nếu chuỗi rỗng
            
        # Thử nhận dạng với độ tin cậy
        langs = detect_langs(cleaned_text)
        
        # Lấy ngôn ngữ có độ tin cậy cao nhất
        if langs and langs[0].prob > 0.5:
            detected_lang = langs[0].lang
            if detected_lang == 'vi':
                return "vi"
            elif detected_lang == 'en':
                return "en"
            
        # Xét các ký tự đặc trưng của tiếng Việt
        vietnamese_chars = set('àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳýỵỷỹđ')
        if any(char.lower() in vietnamese_chars for char in cleaned_text):
            return "vi"
            
        return "en" if re.search(r'[a-zA-Z]', cleaned_text) else "vi"
    except Exception as e:
        print(f"Lỗi nhận dạng ngôn ngữ cho đoạn văn bản: {e}")
        return "vi"  # Mặc định là tiếng Việt nếu có lỗi

def split_text_into_sentences(text):
    """Tách văn bản thành các câu"""
    # Nhận diện mẫu kết thúc câu: dấu chấm, chấm hỏi, chấm than, chấm xuống dòng
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    
    # Xử lý trường hợp có nhiều dấu xuống dòng
    result = []
    for sentence in sentences:
        # Tách theo dấu xuống dòng nếu có
        sub_sentences = re.split(r'\n+', sentence)
        for sub in sub_sentences:
            if sub.strip():
                result.append(sub.strip())
    
    return result

def split_text_by_language(text):
    """Tách văn bản thành các đoạn theo ngôn ngữ"""
    # Tách thành các câu
    sentences = split_text_into_sentences(text)
    
    segments = []
    current_segment = ""
    current_lang = None
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        try:
            # Xác định ngôn ngữ của câu
            sentence_lang = detect_language_for_segment(sentence)
            
            # Nếu đoạn hiện tại trống hoặc cùng ngôn ngữ, thêm câu vào
            if current_lang is None:
                current_lang = sentence_lang
                current_segment = sentence
            elif current_lang == sentence_lang:
                # Thêm dấu cách hoặc dấu chấm nếu cần
                if not current_segment.endswith(('.', '!', '?', '\n')):
                    current_segment += ". "
                else:
                    current_segment += " "
                current_segment += sentence
            else:
                # Nếu khác ngôn ngữ, lưu đoạn hiện tại và bắt đầu đoạn mới
                if current_segment:
                    segments.append((current_segment, current_lang))
                current_segment = sentence
                current_lang = sentence_lang
        except Exception as e:
            print(f"Lỗi khi xử lý câu: {e}")
            # Nếu lỗi, thêm vào đoạn hiện tại nếu có
            if current_segment:
                if not current_segment.endswith(('.', '!', '?', '\n')):
                    current_segment += ". "
                else:
                    current_segment += " "
                current_segment += sentence
            else:
                current_segment = sentence
                current_lang = "vi"  # Mặc định
    
    # Thêm đoạn cuối cùng nếu có
    if current_segment:
        segments.append((current_segment, current_lang))
        
    return segments

def load_model(lang):
    """Tải mô hình theo ngôn ngữ"""
    config = CONFIG[lang]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Đọc config
    preprocess_config = yaml.load(
        open(config['preprocess_config_path'], "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open(config['model_config_path'], "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open(config['train_config_path'], "r"), Loader=yaml.FullLoader
    )
    
    # Tạo pseudo args để tương thích với hàm get_model
    class Args:
        def __init__(self, restore_step):
            self.restore_step = restore_step
    
    args = Args(config['restore_step'])
    
    # Tải model
    model = get_model(
        args,
        (preprocess_config, model_config, train_config),
        device,
        train=False
    )
    
    # Tải vocoder
    vocoder = get_vocoder(model_config, device)
    
    # Đọc stats cho việc hiển thị spectrogram
    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        import json
        stats = json.load(f)
        stats_values = stats["pitch"] + stats["energy"][:2]
    
    # Lưu vào biến toàn cục
    models[lang]['model'] = model
    models[lang]['vocoder'] = vocoder
    models[lang]['preprocess_config'] = preprocess_config
    models[lang]['model_config'] = model_config
    models[lang]['train_config'] = train_config
    models[lang]['stats'] = stats_values
    
    return True

def generate_single_language_speech(text, lang, pitch_control=1.0, energy_control=1.0, duration_control=1.0):
    """Tạo giọng nói sử dụng một ngôn ngữ"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Kiểm tra model đã tải chưa
    if models[lang]['model'] is None:
        success = load_model(lang)
        if not success:
            return None, None
    
    # Chọn config và model
    model = models[lang]['model']
    vocoder = models[lang]['vocoder']
    preprocess_config = models[lang]['preprocess_config']
    model_config = models[lang]['model_config']
    train_config = models[lang]['train_config']
    stats = models[lang]['stats']
    
    # Xử lý text thành phoneme
    if lang == "vi":
        text_sequence = preprocess_vietnamese(text, preprocess_config)
    else:  # English
        text_sequence = preprocess_english(text, preprocess_config)
    
    # Tạo batch giống như trong synthesize.py
    ids = raw_texts = [text[:100]]
    speakers = np.array([0])  # Speaker ID
    texts = np.array([text_sequence])
    text_lens = np.array([len(text_sequence)])
    batch = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
    
    # Tổng hợp giọng nói
    control_values = (pitch_control, energy_control, duration_control)
    
    with torch.no_grad():
        # Convert batch to device
        batch_device = to_device(batch[0], device)
        
        # Chạy model
        output = model(
            *(batch_device[2:]),
            p_control=control_values[0],
            e_control=control_values[1],
            d_control=control_values[2]
        )
        
        # Extract data từ output để tạo spectrogram
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
        
        # Tạo ID duy nhất cho file output
        file_id = str(uuid.uuid4())
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}.wav")
        wavfile.write(output_file, sampling_rate, wav_predictions[0])
        
        # Tạo spectrogram
        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots(1, 1)
        
        pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
        pitch_for_plot = pitch * pitch_std + pitch_mean
        
        im = ax.imshow(mel_prediction.cpu().numpy(), origin="lower", aspect="auto")
        ax.set_title("Spectrogram", fontsize="medium")
        ax.tick_params(labelsize="x-small", left=False, labelleft=False)
        
        ax1 = ax.twinx()
        ax1.plot(pitch_for_plot, color="tomato")
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(labelsize="x-small", colors="tomato")
        
        ax2 = ax.twinx()
        ax2.spines["right"].set_position(("axes", 1.1))
        ax2.plot(energy, color="darkviolet")
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.tick_params(labelsize="x-small", colors="darkviolet")
        
        plt.tight_layout()
        
        # Lưu spectrogram thành file
        spectrogram_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_spec.png")
        plt.savefig(spectrogram_file)
        plt.close()
        
        return output_file, spectrogram_file

def generate_multi_language_speech(text, pitch_control=1.0, energy_control=1.0, duration_control=1.0):
    """Tạo giọng nói đa ngôn ngữ với hiệu ứng chuyển cảnh"""
    # Phân tách văn bản thành các đoạn theo ngôn ngữ
    segments = split_text_by_language(text)
    
    if not segments:
        return None, None
    
    # Tạo ID duy nhất cho output
    file_id = str(uuid.uuid4())
    output_dir = app.config['OUTPUT_FOLDER']
    
    # Tạo danh sách các file âm thanh tạm
    temp_audio_files = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Đảm bảo các model cần thiết đã được tải
    for lang in ['vi', 'en']:
        if any(segment[1] == lang for segment in segments) and models[lang]['model'] is None:
            try:
                load_model(lang)
            except Exception as e:
                print(f"Không thể tải model cho ngôn ngữ {lang}: {e}")
    
    # Tạo âm thanh cho từng đoạn
    for i, (segment_text, lang) in enumerate(segments):
        try:
            print(f"Xử lý đoạn {i+1}/{len(segments)}, ngôn ngữ: {lang}, text: {segment_text[:50]}...")
            
            # Kiểm tra model đã tải chưa
            if models[lang]['model'] is None:
                print(f"Model cho ngôn ngữ {lang} chưa được tải, đang tải...")
                success = load_model(lang)
                if not success:
                    print(f"Không thể tải model {lang}, bỏ qua đoạn này")
                    continue
            
            # Chọn config và model theo ngôn ngữ của đoạn
            model = models[lang]['model']
            vocoder = models[lang]['vocoder']
            preprocess_config = models[lang]['preprocess_config']
            model_config = models[lang]['model_config']
            train_config = models[lang]['train_config']
            
            # Xử lý text thành phoneme
            if lang == "vi":
                # Làm sạch văn bản tiếng Việt trước khi chuyển đổi
                cleaned_text = clean_vietnamese_text(segment_text)
                text_sequence = preprocess_vietnamese(cleaned_text, preprocess_config)
            else:  # English
                text_sequence = preprocess_english(segment_text, preprocess_config)
            
            # Tạo batch
            ids = raw_texts = [segment_text[:100]]
            speakers = np.array([0])  # Speaker ID
            texts = np.array([text_sequence])
            text_lens = np.array([len(text_sequence)])
            batch = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
            
            # Tổng hợp giọng nói
            control_values = (pitch_control, energy_control, duration_control)
            
            with torch.no_grad():
                # Convert batch to device
                batch_device = to_device(batch[0], device)
                
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
                
                temp_file = os.path.join(output_dir, f"{file_id}_segment_{i}.wav")
                wavfile.write(temp_file, sampling_rate, wav_predictions[0])
                temp_audio_files.append((temp_file, sampling_rate))
                print(f"Đã tạo file âm thanh cho đoạn {i+1}: {temp_file}")
                
        except Exception as e:
            print(f"Lỗi tạo audio cho đoạn {i}: {e}")
            continue
    
    if not temp_audio_files:
        print("Không có đoạn nào được xử lý thành công!")
        return None, None
        
    # Ghép các file âm thanh với hiệu ứng fade-in/fade-out
    try:
        print(f"Bắt đầu ghép {len(temp_audio_files)} file âm thanh...")
        combined_audio = None
        
        for i, (temp_file, _) in enumerate(temp_audio_files):
            try:
                segment_audio = AudioSegment.from_wav(temp_file)
                
                # Thêm hiệu ứng fade-in/fade-out
                fade_duration = 100  # 100ms fade
                segment_audio = segment_audio.fade_in(fade_duration).fade_out(fade_duration)
                
                # Thêm khoảng dừng ngắn giữa các đoạn (100ms)
                if i > 0:
                    pause = AudioSegment.silent(duration=100)
                    segment_audio = pause + segment_audio
                
                if combined_audio is None:
                    combined_audio = segment_audio
                else:
                    combined_audio += segment_audio
                    
                print(f"Đã ghép đoạn {i+1}/{len(temp_audio_files)}")
            except Exception as e:
                print(f"Lỗi khi ghép đoạn {i+1}: {e}")
                continue
        
        # Lưu file kết quả
        output_file = os.path.join(output_dir, f"{file_id}_combined.wav")
        combined_audio.export(output_file, format="wav")
        print(f"Đã tạo file âm thanh kết hợp: {output_file}")
        
        # Tạo spectrogram đơn giản cho file kết hợp
        try:
            import librosa
            import librosa.display
            
            # Đọc file kết hợp để tạo spectrogram
            audio, sr = librosa.load(output_file, sr=None)
            plt.figure(figsize=(10, 4))
            
            # Tạo spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            librosa.display.specshow(D, y_axis='log', x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram (Combined Audio)')
            
            # Lưu spectrogram
            spectrogram_file = os.path.join(output_dir, f"{file_id}_spec.png")
            plt.savefig(spectrogram_file)
            plt.close()
            
            print(f"Đã tạo spectrogram: {spectrogram_file}")
            
            # Xóa các file tạm nếu cần
            for temp_file, _ in temp_audio_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
                    
            return output_file, spectrogram_file
        except Exception as e:
            print(f"Lỗi khi tạo spectrogram: {e}")
            return output_file, None
    
    except Exception as e:
        print(f"Lỗi ghép file âm thanh: {e}")
        
        # Nếu không ghép được, sử dụng file cuối cùng làm kết quả
        if temp_audio_files:
            return temp_audio_files[-1][0], None
        
        return None, None

def create_plot_base64(mel, pitch, energy, stats):
    """Tạo spectrogram dạng base64 để hiển thị trên web"""
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch = pitch * pitch_std + pitch_mean
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.imshow(mel, origin="lower", aspect="auto")
    ax.set_title("Spectrogram", fontsize="medium")
    ax.tick_params(labelsize="x-small", left=False, labelleft=False)
    
    ax1 = ax.twinx()
    ax1.plot(pitch, color="tomato")
    ax1.set_ylim(0, pitch_max)
    ax1.set_ylabel("F0", color="tomato")
    ax1.tick_params(labelsize="x-small", colors="tomato")
    
    ax2 = ax.twinx()
    ax2.spines["right"].set_position(("axes", 1.1))
    ax2.plot(energy, color="darkviolet")
    ax2.set_ylim(energy_min, energy_max)
    ax2.set_ylabel("Energy", color="darkviolet")
    ax2.tick_params(labelsize="x-small", colors="darkviolet")
    
    plt.tight_layout()
    
    # Chuyển thành base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_str

def allowed_file(filename):
    """Kiểm tra file có đuôi hợp lệ không"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render trang chủ"""
    vi_model_loaded = models['vi']['model'] is not None
    en_model_loaded = models['en']['model'] is not None
    
    return render_template(
        'index.html', 
        vi_model_loaded=vi_model_loaded,
        en_model_loaded=en_model_loaded
    )

@app.route('/api/generate-speech', methods=['POST'])
def api_generate_speech():
    """API endpoint để tạo giọng nói"""
    try:
        data = request.json
        text = data.get('text', '')
        multi_language_mode = data.get('multi_language_mode', False)
        pitch_control = float(data.get('pitch_control', 1.0))
        energy_control = float(data.get('energy_control', 1.0))
        duration_control = float(data.get('duration_control', 1.0))
        
        if not text:
            return jsonify({"error": "Vui lòng nhập văn bản!"}), 400
        
        # Phát hiện ngôn ngữ
        lang = detect_language(text)
        
        # Kiểm tra model đã tải chưa
        if multi_language_mode:
            if not models['vi']['model'] and not models['en']['model']:
                return jsonify({"error": "Vui lòng tải ít nhất một model trước!"}), 400
        else:
            if not models[lang]['model']:
                return jsonify({"error": f"Vui lòng tải model {lang} trước!"}), 400
        
        # Tạo giọng nói
        if multi_language_mode:
            output_file, spectrogram_file = generate_multi_language_speech(
                text, pitch_control, energy_control, duration_control
            )
        else:
            output_file, spectrogram_file = generate_single_language_speech(
                text, lang, pitch_control, energy_control, duration_control
            )
        
        if not output_file:
            return jsonify({"error": "Không thể tạo giọng nói!"}), 500
        
        # Tạo URL cho file âm thanh và spectrogram
        audio_url = url_for('get_audio', filename=os.path.basename(output_file))
        spectrogram_url = url_for('get_spectrogram', filename=os.path.basename(spectrogram_file)) if spectrogram_file else None
        
        return jsonify({
            "success": True,
            "language": lang,
            "audio_url": audio_url,
            "spectrogram_url": spectrogram_url,
            "multi_language_mode": multi_language_mode
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/load-model', methods=['POST'])
def api_load_model():
    """API endpoint để tải model"""
    try:
        data = request.json
        lang = data.get('language', 'vi')
        
        if lang not in ['vi', 'en']:
            return jsonify({"error": "Ngôn ngữ không hỗ trợ!"}), 400
        
        success = load_model(lang)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Đã tải model {lang} thành công!"
            })
        else:
            return jsonify({
                "error": f"Không thể tải model {lang}!"
            }), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/detect-language', methods=['POST'])
def api_detect_language():
    """API endpoint để phát hiện ngôn ngữ"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "Vui lòng nhập văn bản!"}), 400
        
        lang = detect_language(text)
        
        return jsonify({
            "success": True,
            "language": lang,
            "language_name": "Tiếng Việt" if lang == "vi" else "Tiếng Anh"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/split-text', methods=['POST'])
def api_split_text():
    """API endpoint để phân tách văn bản theo ngôn ngữ"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "Vui lòng nhập văn bản!"}), 400
        
        segments = split_text_by_language(text)
        
        result = []
        for segment_text, lang in segments:
            result.append({
                "text": segment_text,
                "language": lang,
                "language_name": "Tiếng Việt" if lang == "vi" else "Tiếng Anh"
            })
        
        return jsonify({
            "success": True,
            "segments": result,
            "total_segments": len(result)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/audio/<filename>')
def get_audio(filename):
    """Trả về file âm thanh"""
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))

@app.route('/spectrogram/<filename>')
def get_spectrogram(filename):
    """Trả về file spectrogram"""
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))

if __name__ == '__main__':
    # Kiểm tra và tạo thư mục templates nếu chưa tồn tại
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Kiểm tra và tạo thư mục static nếu chưa tồn tại
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Run app
    app.run(host='0.0.0.0', port=5000, debug=True)