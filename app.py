from flask import Flask, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def home():
    return send_from_directory('.', 'view_all_visualizations.html')

@app.route('/saved_models/visualizations/<path:path>')
def serve_visualizations(path):
    return send_from_directory('saved_models/visualizations', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080) 