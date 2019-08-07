from flask import Flask, request, jsonify
import flower_classifier

model = flower_classifier.load_checkpoint('checkpoints/model_vgg16_2048.pth', False)
cat_to_name = flower_classifier.load_cat_to_name('cat_to_name.json')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        content = request.get_json()
        print(content)
        b64 = content['data'].split(',')[1]
        img = flower_classifier.open_image_base64(b64)
        img = flower_classifier.process_image(img)

        print('Predicting...')
        probs, labels = flower_classifier.predict(img, model, 5, False)
        label_names = [cat_to_name[id] for id in labels]
        print('Got result')

        result = []
        for i in range(len(label_names)):
                result.append({ 'name': label_names[i], 'prob': probs[i] })
        print(result)
        return jsonify(result)
    else:
        return 'hi'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')