import os, json
import io, base64
from PIL import Image
import cv2
import numpy as np
np.set_printoptions(suppress=True)
import requests
from io import BytesIO
from keras.models import load_model
import tensorflow as tf

######################## To percent

def to_percent_pipeline(img, mask):
  threshold = .05
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # to grayscale
  img = convert_to_monochrome(img,threshold)
  mask = convert_to_monochrome(mask,threshold)
  severity = calculate_severity(img,mask)
  return severity*100

def calculate_severity(img,mask):
  common = 0
  total = 0

  if img.shape == mask.shape:
    print("OK")
  else:
     print("Mismatch image size when calculating severity")
     
  for row in range(img.shape[0]):
    for col in range(img.shape[1]):
      img_pixel_value = img[row, col]
      mask_pixel_value = mask[row, col]
      if img_pixel_value != 0 and mask_pixel_value != 0:
          common += 1 
      if img_pixel_value != 0:
          total += 1

  print(f"Total : {total}\Common: {common}")

  return common/total


def convert_to_monochrome(image, threshold):
    # Create a copy of the image to avoid modifying the original array
    bw_image = image.copy()

    # Apply thresholding
    bw_image[bw_image < threshold] = 0
    bw_image[bw_image >= threshold] = 1

    return bw_image

########################### end to percent

def extract_coordinates_from_mask(np_mask):
    np_mask = np.array(np_mask).astype(np.uint8) * 255
    # Threshold the mask image to convert it into a binary image
    _, binary_image = cv2.threshold(np_mask, 127, 255, cv2.THRESH_BINARY)
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Extract the coordinates for each contour separately
    coordinates = []
    for contour in contours:
        contour_coordinates = []
        for point in contour:
            x, y = point[0]
            contour_coordinates.append(x)
            contour_coordinates.append(y)
        coordinates.append(contour_coordinates)

    return coordinates

def preprocess_image(image, target_size):
    # Resize the image
    resized_image = cv2.resize(image, target_size)
    # Convert the image to float and rescale the pixel values to [0, 1]
    preprocessed_image = resized_image.astype(np.float32) / 255.0
    # Add an extra dimension to represent a single image
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    return preprocessed_image

def model_inferance(np_arr):
    # file_dir = f"{os.getcwd()}/mysite/models/"
    # file_dir = f"{os.getcwd()}/models/"
    file_dir = "./drive/MyDrive/Project Backup/DRE-Instance-4-Segmentation/"
    path_EX = f"{file_dir}/VGG16_backbone_unet_Instance-EX_Data-imputed_imagenet-Freeze_1024x1024x1_E-1000"
    path_HE = f"{file_dir}/VGG16_backbone_unet_Instance-HE_Data-imputed_imagenet-Freeze_1024x1024x1_E-1000"
    path_MA = f"{file_dir}/VGG16_backbone_unet_Instance-MA_Data-imputed_imagenet-Freeze_1024x1024x1_E-1000"
    path_SE = f"{file_dir}/VGG16_backbone_unet_Instance-SE_Data-imputed_imagenet-Freeze_1024x1024x1_E-1000"

    result = {}
    serverity_value = {"MA": 0, "HE":0, "EX": 0, "SE": 0} 

    if os.path.exists(path_EX):
        model = load_model(path_EX, compile=False)
        mask = np.squeeze(model.predict(np_arr))
        result["EX"] = mask
        serverity_value["EX"] = to_percent_pipeline(np_arr[0],mask)
    else:
        print(f"Invaild model path : {path_EX}")

    if os.path.exists(path_HE):
        model = load_model(path_HE, compile=False)
        mask = np.squeeze(model.predict(np_arr))
        result["HE"] = mask
        serverity_value["HE"] = to_percent_pipeline(np_arr[0],mask)
    else:
        print(f"Invaild model path : {path_HE}")

    if os.path.exists(path_MA):
        model = load_model(path_MA, compile=False)
        mask = np.squeeze(model.predict(np_arr))
        result["MA"] = mask
        serverity_value["MA"] = to_percent_pipeline(np_arr[0],mask)
    else:
        print(f"Invaild model path : {path_MA}")

    if os.path.exists(path_SE):
        model = load_model(path_SE, compile=False)
        mask = np.squeeze(model.predict(np_arr))
        result["SE"] = mask
        serverity_value["SE"] = to_percent_pipeline(np_arr[0],mask)
    else:
        print(f"Invaild model path : {path_SE}")

    return result,serverity_value

def decode_url_to_numpy(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        if img.mode != "RGB":
            img = img.convert("RGB")

        return np.array(img)
    except:
        # Invalid Base64 string
        print("Decode base64 to numpy array failed")
        return None

def resize_to_original_shape(image, target_shape):
    print(f"Resizing to Original Size {target_shape}")
    # Convert the numpy array to PIL Image
    pil_image = Image.fromarray(image)
    # Resize the image using PIL
    resized_image = pil_image.resize(target_shape, resample=Image.NEAREST)
    # Convert the PIL Image back to numpy array
    resized_np_image = np.array(resized_image)

    return resized_np_image


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)  
# CORS(app, origins='*')

# Health check route
@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code

@app.route('/')
def root():
    return f"{request.base_url}-> is Alive. | root dir -> {os.getcwd()} | List dir -> {os.listdir(os.getcwd())} | GPU Status -> {tf.config.list_physical_devices('GPU')}"

@app.route('/api/annotation/v1/', methods=['POST'])
def api():
    response_data = {"annotation":{}}

    data = request.get_json()
    # img_arr = decode_base64_to_numpy(data['encoding'])
    img_arr = decode_url_to_numpy(data['url'])
    img_dims = img_arr.shape
    print(f"Recieved Image Size {img_dims}")

    if img_arr is not None:
        img_arr = preprocess_image(img_arr, (1024,1024))
        masks_dict,severity_dict = model_inferance(img_arr)
        for key, value in masks_dict.items():
            # response_data[key] = extract_coordinates_from_mask(masks_dict[key])
            response_data["annotation"][key] = extract_coordinates_from_mask(resize_to_original_shape(masks_dict[key], target_shape=(img_dims[1], img_dims[0])))
        response_data["severity"] = severity_dict
    else:
        response_data = {"Status": "Invalid Input"}

    ############### SAVE
    with open(f'annotation.json', 'w') as f:
        json.dump(response_data, f ,cls=NumpyEncoder, indent=1)
    ################ LOAD
    f = open('annotation.json', 'rb')
    response_data = json.load(f)

    return jsonify(response_data)


if __name__ == '__main__':
    # app.run(debug=True, host="0.0.0.0", port=8080)
    app.run(debug=True)