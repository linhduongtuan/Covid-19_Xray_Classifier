from flask import Flask, render_template, request
from models import get_Covid_19_Eff_B0, get_tensor, prediction


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'

#model = MyFinalEnsemble_cpu()
model = get_Covid_19_Eff_B0()
model.eval()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')
   

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/inference', methods = ['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        print(request.files)
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file = request.files['file']
        image = file.read()
        top_probs, top_names = prediction(image_bytes=image)
        return render_template('inference.html', top_names=top_names, top_probs=top_probs)

    
if __name__== '__main__':
    print('Loading website')
    print('To open local website, please click on an address at below')
    app.run(debug=True)
 
