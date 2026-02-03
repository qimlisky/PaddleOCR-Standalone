# Compilation instructions
# nuitka-project: --standalone

# nuitka-project: --include-package-data=paddleocr
# nuitka-project: --include-package-data=paddlex
# nuitka-project: --include-package=flask
# nuitka-project: --include-package=numpy
# nuitka-project: --include-package=pandas
# nuitka-project: --include-package=PIL

# nuitka-project-if: {OS} == "Windows":
#     nuitka-project: --output-filename=paddleocr
# nuitka-project-if: {OS} == "Linux":
#     nuitka-project: --output-filename=paddleocr.bin

# nuitka-project: --include-distribution-metadata=imagesize
# nuitka-project: --include-distribution-metadata=opencv-contrib-python
# nuitka-project: --include-distribution-metadata=pyclipper
# nuitka-project: --include-distribution-metadata=pypdfium2
# nuitka-project: --include-distribution-metadata=shapely
# nuitka-project: --include-data-dir=D:\AI\OCR\new\runtime\Lib\site-packages\paddle\libs=paddle\libs


# Windows-specific metadata for the executable
# nuitka-project-if: {OS} == "Windows":
#     nuitka-project: --file-description="PaddleOCR Standalone Executable"
#     nuitka-project: --file-version="1.3.2"
#     nuitka-project: --product-name="PaddleOCR-GPU"
#     nuitka-project: --product-version="1.3.2"
#     nuitka-project: --copyright="timminator"
#     nuitka-project: --windows-icon-from-ico=paddleocr.ico



from flask import Flask, request, jsonify
import re
import io
import numpy as np
from PIL import Image
import os
import sys
import traceback
from paddleocr.__main__ import console_entry

# --- Crucial: Set cache directory relative to the executable ---
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ["PADDLE_PDX_CACHE_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".paddlex")

app = Flask(__name__)

print(f"Python: {sys.version.split()[0]} | Initializing PaddleOCR...")
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    print(" Model loaded successfully.")
    model_loaded = True
except Exception as e:
    print(f"Initialization failed: {e}")
    traceback.print_exc()
    ocr = None
    model_loaded = False

def extract_text_from_predict_result(result):
    texts = []
    try:
        if not result or not isinstance(result, list):
            return ""
        page = result[0]
        if hasattr(page, 'json') and isinstance(page.json, dict):
            res_data = page.json.get('res', {})
            rec_texts = res_data.get('rec_texts', [])
            
            if isinstance(rec_texts, list):
                for text in rec_texts:
                    if text and str(text).strip():
                        texts.append(str(text).strip())
            else:
                print(f"[Warning] rec_texts is not a list: {type(rec_texts)}")
        else:
            print(f"[Warning] No 'json' attribute: {type(page)}")
    except Exception as e:
        print(f"[Extraction Error] {e}")
        traceback.print_exc()
    return '\n'.join(texts)

@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
    if not model_loaded:
        return jsonify({'success': False, 'error': 'OCR service is not ready.'}), 503

    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'Missing image parameter.'}), 400

        file = request.files['image']
        mode = request.form.get('mode', 'clean').strip().lower()

        if not file.filename:
            return jsonify({'success': False, 'error': 'Filename is empty.'}), 400

        img_bytes = file.read()
        if not img_bytes:
            return jsonify({'success': False, 'error': 'Image content is empty.'}), 400

        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_np = np.array(img)

        result = ocr.predict(input=img_np)
        raw_text = extract_text_from_predict_result(result)

        if mode == "raw":
            text = raw_text
        elif mode == "chinese":
            text = re.sub(r'[^\u4e00-\u9fa5\s\n]', '', raw_text)
            text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
        else: # clean (default)
            text = re.sub(
                r'[^\u4e00-\u9fa5a-zA-Z0-9\s\u3000-\u303f\uff00-\uffef，。！？；：（）【】《》、·\-,.!?;:\'\"/\\]',
                '', raw_text
            )
            text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())

        return jsonify({'success': True, 'text': text if text else ""})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Processing error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'text_extraction': "page.json['res']['rec_texts']",
        'python': sys.version.split()[0],
        'cache_dir': os.environ.get("PADDLE_PDX_CACHE_HOME", "Not Set")
    })

# if __name__ == '__main__':
#     if model_loaded:
#         print("\n" + "="*60)
#         print("PaddleOCR initialized successfully.")
#         print("💡 Text extraction method: result[0].json['res']['rec_texts']")
#         print("💡 Cache directory set to local '.paddlex' folder.")
#         print("📍 Test: curl -X POST http://localhost:5000/api/ocr -F \"image=@test.png\"")
#         print("Starting API service on http://0.0.0.0:5000")
#         print("="*60)
#         app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
#     else:
#         print("❌ Service startup failed due to model loading error.")
#         sys.exit(1)
def main():
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断 (Ctrl+C)")
    except Exception as e:
        # 打印详细的错误信息
        import traceback
        print("程序发生未捕获的异常:")
        traceback.print_exc() # 这会打印完整的堆栈跟踪
        print("\n按 Enter 键退出...")
        input() # 等待用户按键后才退出
    if len(sys.argv) == 1: 
        sys.argv.append("--help") 
    console_entry()    