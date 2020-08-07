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
import math
import cv2




import path


app = Flask(__name__)

@app.route('/')
def index():
    img_path = ""
    data = {}

    cat_temperament = set()
    for temp in df_cat_t.Temperment: 
        if isinstance(temp, list):
            for t in temp:
                cat_temperament.add(t)  

    for temp in df_dog_t.Temperment: 
        if isinstance(temp, list):
            for t in temp:
                cat_temperament.add(t) 

    price = set()
    for temp in df_cat_t.AvgKittenPrice: 
        if isinstance(temp, float):
            price.add(temp)
    for temp in df_dog_t.AvgPupPrice: 
        if isinstance(temp, float):
            price.add(temp)

    cleanlist = [0.0 if math.isnan(x) else x for x in price]

    wt = set()
    for temp in df_cat_t.MaleWtKg: 
        if isinstance(temp, float):
            wt.add(temp)
    for temp in df_dog_t.MaleWtKg: 
        if isinstance(temp, float):
            wt.add(temp)
    wt_list = [0.0 if math.isnan(x) else x for x in wt]



    data['price'] = sorted(set(cleanlist))
    data['temper'] = sorted(cat_temperament)
    data['weight'] = sorted(set(wt_list))
    return render_template('index.html', imagepath = img_path, data = data)

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

cat_breeds = ['abyssinian','bengal','birman','bombay','british_shorthair','egyptian_mau','maine_coon','persian','ragdoll','russian_blue','siamese','sphynx']

df_cat = pd.read_csv("C:/Users/manis/Downloads/datasets_144588_337971_cat_breed_characteristics.csv")
df_cat_t = df_cat.apply(lambda x: x.str.split(', ') if x.name == 'Temperment' else x)
df_dog_t = df_dog.apply(lambda x: x.str.split(', ') if x.name == 'Temperment' else x)

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
data_cat = ImageDataBunch.from_name_re(path_img, fnames_cat, pat, ds_tfms=get_transforms(), size=224, bs=bs).normalize(imagenet_stats)

# data_aug_dog = ImageDataBunch.from_folder(path_img,fnames_dog, ds_tfms=tfms, size=224, bs=bs).normalize(imagenet_stats)

learn_dog = cnn_learner(data_aug_dog, models.resnet50, metrics=accuracy)           
learn_cat = cnn_learner(data_cat, models.resnet50, metrics=accuracy)           


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def is_human(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces)
    
@app.route("/checkbreed-section")
def predict_breed_dog(img_path,learn_dog, category):
    res_human = is_human(img_path)
    if res_human > 0:
        new1 = {}
        s = img_path.split("/")
        print(s[-1:])
        img_path = 'images/' + str(s[-1:][0])
        return render_template("breed_info.html", data = new1 , imagepath = img_path, breedname = "Human",category = "Human")

    test_img =  open_image(img_path)
    test_img.show()
    learn_dog = learn_dog.load('C:/Users/manis/Downloads/pet_breed_classification/static/images/models/model_resnet_dog')
    breed_idx = int(learn_dog.predict(test_img)[0])
    char = infoTable(df_dog,data_dog1[breed_idx])
    s = img_path.split("/")
    t = s[-1:][0].split("/")
    print(t[-1:])
    img_path = 'images/' + str()
    # return data_dog1[breed_idx]
    if char.shape[1] == 2:
        new1 = {}
        for i, row in char.iterrows():
            new1[row[0]]= row[1]
        return render_template("breed_info.html", data = new1, imagepath = img_path, breedname = data_dog1[breed_idx],category = category)
    return render_template("breed_info.html", data = char, imagepath = img_path, breedname = data_dog1[breed_idx],category =  category)

def predict_breed_cat(img_path,learn_cat, category):
    res_human = is_human(img_path)
    if res_human > 0:
        new1 = {}
        s = img_path.split("/")
        print(s[-1:])
        img_path = 'images/' + str(s[-1:][0])
        return render_template("breed_info.html", data = new1 , imagepath = img_path, breedname = "Human",category = "Human")

    test_img =  open_image(img_path)
    test_img.show()
    learn_cat = learn_cat.load('C:/Users/manis/Downloads/pet_breed_classification/static/images/models/model_resnet_cat')
    breed_idx = int(learn_cat.predict(test_img)[0])
    char = infoTable(df_cat,cat_breeds[breed_idx])
    s = img_path.split("/")
    print(s[-1:])
    img_path = 'images/' + str(s[-1:][0])
    # return data_dog1[breed_idx]
    if char.shape[1] == 2:
        new1 = {}
        for i, row in char.iterrows():
            new1[row[0]]= row[1]
        return render_template("breed_info.html", data = new1, imagepath = img_path, breedname = cat_breeds[breed_idx],category = category)
    return render_template("breed_info.html", data = char, imagepath = img_path, breedname = cat_breeds[breed_idx],category =  category)


@app.route("/dataandeda")
def dataandeda():
    if request.method == "POST":
        return render_template("datasetandeda.html")
    return render_template("datasetandeda.html")



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
            category = request.form['category']

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
            if category == 'dog':
                breed = predict_breed_dog(img_paths,learn_dog, category)
            if category == 'cat':
                breed = predict_breed_cat(img_paths,learn_cat, category)
    return breed




def findbreed(data, min_price=0, max_price=5000, temperament=[], min_wlt=0,max_wlt=90):
  target = pd.DataFrame()
  if data=='cat':
    df =df_cat_t
    df = df[df['MaleWtKg']>=min_wlt]
    df = df[df['MaleWtKg']<=max_wlt]
    df = df[df['AvgKittenPrice']>=min_price]
    df = df[df['AvgKittenPrice']<=max_price]
    for i in range(len(df)):
      if set(temperament) <= set(df['Temperment'].iloc[i]):
        target=target.append(df[i:i+1],ignore_index = True)
    return target
  elif data== 'dog':
    df =df_dog_t
    df = df[df['MaleWtKg']>=min_wlt]
    df = df[df['MaleWtKg']<=max_wlt]
    df = df[df['AvgPupPrice']>=min_price]
    df = df[df['AvgPupPrice']<=max_price]
    for i in range(len(df)):
      if set(temperament) <= set(df['Temperment'].iloc[i]):
        target=target.append(df[i:i+1],ignore_index = True)
    return target
  else:
    print("please input 'cat' or 'dog")
    return target

@app.route('/checktemparament',  methods=["GET", "POST"])
def checktemparament():

    temperament = []
    category = 'cat'
    category = request.form['category']
    temperament = request.form.getlist('temperament')
    min_price, max_price = 0,3000
    min_price =  request.form['min_price']
    max_price =  request.form['max_price']

    min_weight, max_weight = 0,90
    min_weight =  request.form['min_weight']
    max_weight =  request.form['max_weight']
    # min_price=0, max_price=5000, temperament=[], min_wlt=0,max_wlt=90
    res=findbreed(category,float(min_price),float(max_price),temperament,float(min_weight),float(max_weight))
    print(res)

    
    # data = res
    return render_template("breed_temperament.html", data = res, category = category)


if __name__ == '__main__':
    
    app.run(debug=True)
    




