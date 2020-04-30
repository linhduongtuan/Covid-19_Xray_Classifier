from flask import Flask, request, render_template
app = Flask(__name__)

from commons import get_tensor
from inference import prediction

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html', value='hi')
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file = request.files['file']
        image = file.read()
        top_probs, top_labels, top_diseases = prediction(image_bytes=image)
        prediction(image_bytes=image)
        tensor = get_tensor(image_bytes=image)
        print(get_tensor(image_bytes=image))
        return render_template('result.html', disease=top_diseases, probabilities=top_probs)

if __name__ == '__main__':
	app.run(debug=True)
