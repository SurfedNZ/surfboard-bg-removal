from flask import Flask, request, send_file
from u2net_test import remove_background
import io

app = Flask(__name__)

@app.route('/remove-background', methods=['POST'])
def remove_bg():
    if 'image' not in request.files:
        return {"error": "No image uploaded"}, 400

    image = request.files['image']
    output_image = remove_background(image)
    img_io = io.BytesIO()
    output_image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)