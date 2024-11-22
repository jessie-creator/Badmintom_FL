from flask import Flask, request
from flask import jsonify
from flask import send_file
import subprocess
import uuid

app = Flask(__name__)

used_uuids = set()
with open('uuid.txt', 'r') as file:
    for line in file:
        uuid_str = line.strip()
        used_uuids.add(uuid_str)
def generate_unique_uuid():
    new_uuid = str(uuid.uuid4())
    while new_uuid in used_uuids:
        new_uuid = str(uuid.uuid4())
    used_uuids.add(new_uuid)
    return new_uuid

@app.route('/create_uuid', methods=['GET'])
def create_uuid():
    new_uuid = generate_unique_uuid()
    used_uuids.add(new_uuid)
    with open('uuid.txt', 'a') as file:
        file.write(new_uuid + '\n')
    return jsonify({"uuid": new_uuid})

@app.route('/download_model', methods=['GET'])
def download_model():
    model_type = request.args.get('model_type')  # 從 GET 請求的參數中獲取 model_type

    # 檢查 model_type 是否為有效值，根據不同的值下載相應的模型檔案
    if model_type == 'high':
        model_filename = 'uploaded_high_model.pth'  # 設定高級模型的檔案路徑和名稱
    elif model_type == 'short':
        model_filename = 'uploaded_short_model.pth'  # 設定短期模型的檔案路徑和名稱
    else:
        return "無效的模型類型", 400

    return send_file(model_filename, as_attachment=True)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    uuid = request.args.get('uuid')
    if 'high_model_file' not in request.files or 'short_model_file' not in request.files:
        print("未收到全部模型檔案, 400")
        return "未收到全部模型檔案", 400
    
    high_model_file = request.files['high_model_file']
    short_model_file = request.files['short_model_file']
    print(high_model_file.filename)
    print(short_model_file.filename)

    # 儲存上傳的檔案
    if high_model_file.filename != '' and short_model_file.filename != '':
        if uuid:
            high_model_file.save("model_from_clients/high/" + uuid + "_model.pth")#_file.filename)#'uploaded_high_model.pth')"model_from_clients/high/" + 
            short_model_file.save("model_from_clients/short/" + uuid + "_model.pth")#_file.filename)#'uploaded_short_model.pth')"model_from_clients/short/" + 
            return "模型檔案已成功上傳", 200

    return "檔案名稱錯誤", 400


@app.route('/run_backend_process', methods=['POST'])
def run_backend_process():
    try:
        subprocess.run(['python3', 'fl.py'])  # 執行 fl.py 檔案
        return jsonify({"message": "後端程式已啟動"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
