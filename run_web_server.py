import os
import sys
import argparse
import socket
from tts_web_app import app

def get_ip_address():
    """Lấy địa chỉ IP của máy chủ"""
    try:
        # Tạo kết nối giả để lấy địa chỉ IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"  # Mặc định nếu không lấy được

def main():
    """
    Khởi động máy chủ web Text-to-Speech
    """
    parser = argparse.ArgumentParser(description='Khởi động máy chủ web Text-to-Speech')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Địa chỉ máy chủ')
    parser.add_argument('--port', type=int, default=8080, help='Cổng máy chủ')
    parser.add_argument('--debug', action='store_true', help='Chế độ debug')
    
    args = parser.parse_args()
    
    # Lấy địa chỉ IP thực tế
    ip_address = get_ip_address()
    
    # Hiển thị URL truy cập
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║    🎯 Máy chủ Text-to-Speech đang chạy!                         ║
║                                                                  ║
║    - Truy cập tại địa chỉ cục bộ:   http://127.0.0.1:{args.port}       ║
║    - Truy cập từ các thiết bị khác: http://{ip_address}:{args.port}     ║
║                                                                  ║
║    ℹ️ Vui lòng đảm bảo tường lửa cho phép kết nối đến cổng {args.port}  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Khởi động máy chủ Flask
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    # Kiểm tra xem các thư mục cần thiết đã tồn tại chưa
    if not os.path.exists('templates'):
        print("Thư mục 'templates' không tồn tại. Đang tạo...")
        os.makedirs('templates')
    
    if not os.path.exists('static'):
        print("Thư mục 'static' không tồn tại. Đang tạo...")
        os.makedirs('static')
    
    if not os.path.exists('output/result/web_output'):
        print("Thư mục 'output/result/web_output' không tồn tại. Đang tạo...")
        os.makedirs('output/result/web_output', exist_ok=True)
    
    main() 