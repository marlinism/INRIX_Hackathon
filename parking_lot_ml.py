import Roboflow from roboflow

# this func use the model to calculate the percentage of space-empty in a parking lot
def predict_percentage(model, img):
    a = model.predict(img, confidence=40, overlap=30).json()
    perc = 0
    for i in range(a['predictions']):
        if i['class'] == 'space-empty':
            perc += 1
    return perc / len(a['predictions'])

rf = Roboflow(api_key="rf_l8XtjkeiHQUPlsFmBGfKb6DpS393") # publishable api key
project = rf.workspace().project("pklot-1tros")
model = project.version(2).model

img = ""
print(predict_percentage(model, img))