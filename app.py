from flask import Flask, request
from flask import jsonify
import base64
import os
import openai
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from pprint import pprint
import json
# # Create a new YOLO model from scratch
# model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
# model = YOLO('yolov8n.pt')
model = YOLO('yolov8n_openvino_model/')
# # Export the model
# model.export(format='openvino')
def make_request(question_input: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{question_input}"},
        ]
    )
    return response
openai.api_key = os.getenv("OPENAI_API_KEY")
print("openai.api_key: ",openai.api_key)
start_sequence = "\nAI:"
restart_sequence = "\nHuman: "
# import logging
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)


atob = lambda x:base64.b64decode(x)
btoa = lambda x:base64.b64encode(bytes(x, 'utf-8')).decode('utf-8')

app = Flask(__name__)

@app.route('/data', methods=['GET'])
def get_query_string():
    # return request.query_string
    prompt=request.args.get('prompt').encode('latin-1')
    decoded = atob(prompt)
    im = Image.open(BytesIO(base64.b64decode(prompt)))
    print(im.size, )

    im.save("saved_img.png")
    # results = model(im)  # list of Results objects
    results = model.predict('saved_img.png', save=True)
    # print(results.tojson())
    json_d = {}
    json_d['predictions'] = []
    for ind,result in enumerate(results):
        # print(result.tojson())
        for b in result.boxes:
            # f'{name} {conf:.2f}'
            cls_ind = int(b.cls[0])
            print(ind,b.xyxy.cpu().numpy()[0],cls_ind,result.names[cls_ind], b.conf.item())
            json_d['predictions'].append({
                'box':[int(i) for i in b.xyxy.cpu().numpy()[0]],
                'class': result.names[cls_ind],
                'conf': b.conf.item()
            })
    pprint(json_d)
    #     boxes = result.boxes  # Boxes object for bbox outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     probs = result.probs  # Class probabilities for classification outputs
    #     print()
    # print("original: ",prompt)
    prompt = f'''Generate only an informative and nature paragraph based on the given json:\n{json.dumps(json_d)}\nHere are some rules:\nShow object and and position.\nUse nouns rather than coordinates to show position information of each object. No more than 50 characters. Describe position of each object.Do not appear number. Do not say labeled, do not say confidence.\n'''
    # print(prompt)
    # prompt="What is AI?"
    # print("decoded: ",decoded)
    # response = openai.Completion.create(
    #     model="text-davinci-003",
    #     prompt=prompt,
    #     temperature=0.9,
    #     max_tokens=150,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0.6,
    #     stop=[" Human:", " AI:"]
    #     )
    response = make_request(prompt)
    print(response)
    # return {"prompt": response['choices'][0]['text'][4:]}
    return {"prompt": response["choices"][0]["message"]["content"]}
    # return {"prompt": 'test'}

if __name__ == '__main__':
    app.run(debug=False,port=3000,host="0.0.0.0")
