from flask import Flask, render_template, request, redirect
import io
import os
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
from fastai.vision import *
from fastai.metrics import error_rate,accuracy
import base64

import path


app = Flask(__name__)

@app.route('/')
def index():
    img_path = ""
    return render_template('index.html', imagepath = img_path)

app.config["UPLOAD_FOLDER"] = "C:/Users/manis/Downloads/pet_breed_classification/static/images/"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF",  'png', 'jpg', 'jpeg']

df_dog = pd.read_csv("C:/Users/manis/Downloads/datasets_144588_337971_dog_breed_characteristics.csv")
data_dog1 = ['american_bulldog',
 'american_pit_bull_terrier',
 'basset_hound',
 'beagle',
 'boxer',
 'chihuahua',
 'english_cocker_spaniel',
 'english_setter',
 'german_shorthaired',
 'great_pyrenees',
 'havanese',
 'japanese_chin',
 'keeshond',
 'leonberger',
 'miniature_pinscher',
 'newfoundland',
 'pomeranian',
 'pug',
 'saint_bernard',
 'samoyed',
 'scottish_terrier',
 'shiba_inu',
 'staffordshire_bull_terrier',
 'wheaten_terrier',
 'yorkshire_terrier']

bs = 32
#help(untar_data)
path = untar_data(URLs.PETS)
path_anno = path/'annotations'
path_img = path/'images'

np.random.seed(1)
pat = r'/([^/]+)_\d+.jpg$'


fnames = get_image_files(path_img)

fnames_cat=[]
fnames_dog=[]
for fn in fnames:
  if fn.name.lower().startswith(tuple(['abyssinian','bengal','birman','bombay','british_shorthair','egyptian_mau','maine_coon','persian','ragdoll','russian_blue','siamese','sphynx'])):
    fnames_cat.append(fn)
  else:
    fnames_dog.append(fn)
tfms = get_transforms(do_flip=True)
# print(path_img)
# print(fnames_dog)

data_aug_dog = ImageDataBunch.from_name_re(path_img, fnames_dog, pat, ds_tfms=get_transforms(flip_vert=True),size=224, bs=bs).normalize(imagenet_stats)
# data_aug_dog = ImageDataBunch.from_folder(path_img,fnames_dog, ds_tfms=tfms, size=224, bs=bs).normalize(imagenet_stats)

learn_dog = cnn_learner(data_aug_dog, models.resnet50, metrics=accuracy)           
                       
@app.route("/checkbreed-section")
def predict_breed(img_path,learn_dog):
    test_img =  open_image(img_path)
    test_img.show()
    learn_dog = learn_dog.load('C:/Users/manis/Downloads/pet_breed_classification/static/images/models/model_resnet_dog')
    breed_idx = int(learn_dog.predict(test_img)[0])
    char = infoTable(df_dog,data_dog1[breed_idx])
    s = img_path.split("/")
    print(s[-1:])
    img_path = 'images/' + str(s[-1:][0])
    # return data_dog1[breed_idx]
    if char.shape[1] == 2:
        new1 = {}
        for i, row in data.iterrows():
            new1[row[0]]= row[1]
        return render_template("breed_info.html", data = new1, imagepath = img_path)
    
    return render_template("breed_info.html", data = char, imagepath = img_path)



def allowed_image(filename):

    # We only want files with a . in the filename
    if not "." in filename:
        return False

    # Split the extension from the filename
    ext = filename.rsplit(".", 1)[1]

    # Check if the extension is in ALLOWED_IMAGE_EXTENSIONS
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

def infoTable(df,breed):
    l=breed.strip().lower().split()
    breed=""
    for w in l:
        breed+=w.capitalize()+" " 
        breed = breed.strip()
        if df[df['BreedName']==breed].empty and df[df['AltBreedName']==breed].empty:
            print("no information of "+breed+" found.")
            return df[df['BreedName']==breed].T
        elif df[df['BreedName']==breed].empty:
            return df[df['AltBreedName']==breed].T
        else:
            return df[df['BreedName']==breed].T



@app.route("/checkbreed", methods=["GET", "POST"])
def upload_image():
    breed = ""

    if request.method == "POST":
        if request.files:

            image = request.files["file"]
            global img_path 
            img_path =  image.filename
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], image.filename))
            # if image.filename == "":
            #     print("No filename")
            #     return redirect(request.url)
            # if allowed_image(image.filename):
            #     filename = secure_filename(image.filename)

            #     image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

            #     print("Image saved")
            #     

            #     return redirect("https://localhost:5000/")

            # else:
            #     print("That file extension is not allowed")
            #     return redirect("https://localhost:5000/")    

            # print(image)
            # image.save(os.path.join(app.config["UPLOAD_FOLDER"], image))
            img_paths = "C:/Users/manis/Downloads/pet_breed_classification/static/images/" + str(image.filename)
            breed = predict_breed(img_paths,learn_dog)
    return breed

if __name__ == '__main__':
    
    app.run(debug=True)
    




