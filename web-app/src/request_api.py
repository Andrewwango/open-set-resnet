import requests

HOSTNAME = 'http://open-set-resnet-api.herokuapp.com'
#HOSTNAME = 'http://localhost:8000'

API_URL = HOSTNAME + '/prediction/'
LIST_URL = HOSTNAME + '/list_model_names/'
STATIC_TEXT_URL = HOSTNAME + '/static_text/'

def get_model_options():
    return requests.get(LIST_URL).json()["model_names"]

def post_image_for_inference(bytes=None, model=None, open_set=None):
    response = requests.post(API_URL, files={"file" : bytes}, 
                                    params={"model_name" : model, "open_set" : open_set})
    results = response.json()
    return results

def get_static_text(model_name=None):
    return requests.get(STATIC_TEXT_URL, params={"model_name" : model_name}).json()["static_text"]