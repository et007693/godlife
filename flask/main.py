from flask import Flask, request,  jsonify
from ultralytics import YOLO

import random
import os

app = Flask(__name__)

object = ''
object_id = ''

@app.route('/')
def home():
  return '갓생루팅'

@app.route('/api/v1/mission/object/', methods=['GET', 'POST'])
def object_detect():
  global object
  if request.method == 'GET':
    object_list = ['monitor', 'refrigerator', 'pen', 'toothbrush', 'chopstick', 'towel', 'book','toothpaste']
    object_id = random.randint(0, len(object_list) - 1)
    object = object_list[object_id]

    response = {
        "data": {
            "object_id": str(object_id + 1),
            "object_name": object
        },
        "message": "실내 미션 조회에 성공하였습니다."
    }
    return jsonify(response)
  
  if request.method == 'POST':
    model = YOLO(f'./model/{object}.pt')
    file = request.files['file']
    file.save('./temp/' + file.filename)
    train_img = './temp/' + file.filename
    result = model(train_img)
    os.remove('./temp/' + file.filename)

  if result[0].boxes:
    if max(result[0].boxes.conf) > 0.5:
      confidence = round(max(result[0].boxes.conf).item(), 2)
      print(confidence)
      success = True
    else:
      confidence = [round(max(result[0].boxes.conf).item(), 2)]
      success = False
  else:
    confidence = 0
    success = False

  response = {
      "data": {
          "object_id": str(object_id + 1),
          "object_name": object
      },
      "detect": {
        "confidence": confidence,
        "success": success
      }
  } 

  return jsonify(response)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000, debug=True)
