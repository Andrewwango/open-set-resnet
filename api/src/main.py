from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel
import io, sys
import open_set_inference as osi
import assetloader, inference

app = FastAPI()

class Prediction(BaseModel):
  filename: str
  contenttype: str
  pred: str
  conf: str
  level: int

@app.get('/')
def root_route():
  return { 'error': 'Use GET /prediction instead of the root route!' }

@app.get('/list_model_names/')
def get_model_names():
  return {"model_names": assetloader.get_model_names()}

@app.get('/static_text/')
def get_static_text(model_name):
  #model name param
  return {"static_text": assetloader.get_static_text(model_name=model_name)}

@app.post('/prediction/', response_model=Prediction)
async def prediction_route(file: UploadFile = File(...), model_name: str = "", open_set: bool = True):

  # Ensure that this is an image
  #if file.content_type.startswith('image/') is False:
  #  print("ohno")
  #  raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

  try:
    # Read image contents
    contents = await file.read()
    image = io.BytesIO(contents)
    # Generate prediction
    if open_set:
      results = osi.classify_open_set(image=image, model_name=model_name)
      level = results[-1]
    else:
      results = inference.classify_breeds(image=image, display_image=False, model_name=model_name)
      level = 2
    return {
      'filename': file.filename,
      'contenttype': file.content_type,
      'pred': results[0][0],
      'conf': results[1][0],
      'level': level
    }
  except:
    e = sys.exc_info()[1]
    print(e)
    raise HTTPException(status_code=500, detail=str(e))