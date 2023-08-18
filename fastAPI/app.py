try:
    import os
    import uuid
    import json
    import shutil
    import base64
    import logging

    from PIL import  Image
    from io import BytesIO
    from typing import Union, Any
    from pydantic import BaseModel
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi import FastAPI, File, UploadFile, HTTPException

    from detect import run
except Exception as e:
    print('[ERROR] Load error in app.py: ', e)

EXPORTED_MODELS_DIR = "path/to/exported/models"
conf_threshold = 0.6

app = FastAPI()

# Pydantic models 
class ImageJson(BaseModel):        # model is used to represent the JSON response containing information about detected objects in an image.
    label: Any                 

class BASE64Input(BaseModel):      # model is used to accept base64-encoded image data as input to the inference APIs
    image_path: str


def remove_dir(dir_path="runs/detect/exp"):
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Directory {dir_path} removed successfully.")
        except OSError as e:
            print(f"Error: {dir_path} : {e.strerror}")
    else:
        print(f"Directory {dir_path} does not exist.")

def base64_to_image(base64_str, save_path="images_from_base64\image.png", format="PNG"):
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    img.save(save_path, format=format)
    return img

def txt_to_json(txt, image_width=1280, image_height=720):
    lines = txt.strip().split("\n")
    json_objects = []

    for i, line in enumerate(lines):
        class_id, x_center, y_center, width, height, acc = map(float, line.split())

        left = (x_center - width / 2) * image_width
        right = (x_center + width / 2) * image_width
        top = (y_center - height / 2) * image_height
        bottom = (y_center + height / 2) * image_height

        object_class_name = "Wheel" if class_id == 1 else "Dolly"

        json_object = {
            "Id": i,
            "ObjectClassName": object_class_name,
            "ObjectClassId": int(class_id),
            "Left": round(left),
            "Top": round(top),
            "Right": round(right),
            "Bottom": round(bottom),
            "Accuracy": acc,
        }

        json_objects.append(json_object)

    json_output = json.dumps(json_objects, indent=4)
    return json_output



@app.get("/")
def read_root():
    return {"message": "Welcome to the Inference API!"}

@app.get("/models")
def list_models():
    available_models = []

    model_files = os.listdir(EXPORTED_MODELS_DIR)

    onnx_model_files = [file for file in model_files if file.endswith(".onnx")]

  
    model_names = [model_file[:-5] for model_file in onnx_model_files]

    available_models = model_names

    return {"models": available_models}

@app.get("/model_labels/{model_name}")
def get_model_labels(model_name: str):
    model_path = os.path.join(EXPORTED_MODELS_DIR, f"{model_name}.onnx")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    labels = ["Wheel", "Dolly"] # always same labels

    return {"model": model_name, "labels": labels}

@app.post("/{model_name}/json/", response_model=ImageJson)
def get_json(item: BASE64Input, model_name: str):
    remove_dir("runs/detect/exp")  # remove results from previous inference
    base64_to_image(item.image_path)

    if model_name == "yolov5":
        run(
            weights="runs/train/exp/weights/yolov5.onnx",
            conf_thres=conf_threshold,
            source="images_from_base64",
            save_txt=True,
            save_conf=True,
            name="exp",
        )

    elif model_name == "detr":
        run(

            )

    else:
        raise HTTPException(status_code=404, detail="Model Not Found !")

    with open(r"runs/detect/exp/labels/image.txt", "r") as f:
        txt = f.read()

    json_str = txt_to_json(txt)
    objects = json.loads(json_str)
    remove_dir("runs/detect/exp")
    return {"objects": objects}

@app.post("/{model_name}/predict-box/")
async def predict_box(model_name: str, file: UploadFile = File(...)):
    file.filename = f"image.png"
    contents = await file.read()
    remove_dir("runs/detect/exp")

    with open(f"data/images/{file.filename}", "wb") as f:
        f.write(contents)

    if model_name == "yolov5":
        run(
            weights="runs/train/exp/weights/yolov5.onnx",
            conf_thres=conf_threshold,
            source="data/images",
            save_txt=True,
            save_conf=True,
            name="exp",
        )

    elif model_name == "dtr":
        run(
            
        )

    else:
        raise HTTPException(status_code=404, detail="Model Not Found !")

    return FileResponse("runs/detect/exp/image.png")

@app.post("/{model_name}/predict-json/", response_model=ImageJson)
async def predict_json(model_name: str, file: UploadFile = File(...)):
    file.filename = f"image.png"
    contents = await file.read()
    remove_dir("runs/detect/exp")

    with open(f"data/images/{file.filename}", "wb") as f:
        f.write(contents)

    if model_name == "yolov5":
        run(
            weights="runs/train/exp/weights/yolov5s.onnx",
            conf_thres=conf_threshold,
            source="data/images",
            save_txt=True,
            save_conf=True,
            name="exp",
        )

    elif model_name == "detr:
        run(
            
        )

    else:
        raise HTTPException(status_code=404, detail="Model Not Found !")

    with open(r"runs/detect/exp/labels/image.txt", "r") as f:
        txt = f.read()

    json_str = txt_to_json(txt)
    label = json.loads(json_str)
    remove_dir("runs/detect/exp")
    return {"label": label}

