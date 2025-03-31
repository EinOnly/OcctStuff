from flask import Flask, request, jsonify
from utilities.log import CORELOG
import requests

app = Flask(__name__)

@app.route('/api/test', methods=['POST'])
def test_api():
    url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=976640da-7aab-47d7-b035-59e105dfabd2'
    headers = {'Content-Type': 'application/json'}
    data = request.get_json()

    response = requests.post(url, headers=headers, json=data)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)