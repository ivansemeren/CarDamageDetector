from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from carapp.models import PicUpload
from carapp.forms import ImageForm
# Create your views here.

def index(request):
    return render(request, 'index.html')

def list(request):
    image_path = ''
    image_path1 = ''

    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        
        if form.is_valid():
            newdoc = PicUpload(imagefile=request.FILES['imagefile'])
            newdoc.save()

            return HttpResponseRedirect(reverse('list'))

    else:
        form = ImageForm()  

    documents = PicUpload.objects.all()

    for document in documents:
         image_path = document.imagefile.name
         image_path1 = '/' + image_path
         document.delete()

    request.session['image_path'] = image_path
    
    return render(request, 'list.html', {'documents': documents, 'image_path': image_path1, 'form':form })


import os
import json

import h5py
import numpy as np
import pickle as pk
from PIL import Image


from keras.preprocessing.image import img_to_array, load_img
from keras.utils.data_utils import get_file
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
from keras import backend as K
import tensorflow as tf

def prepare_img_224(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

car_list = [('n03770679', 'minivan'),('n04285008', 'sports_car'),('n03100240', 'convertible'),('n02814533', 'beach_wagon'),('n03930630', 'pickup'),
 ('n03670208', 'limousine'),('n04037443', 'racer'),('n03594945', 'jeep'),('n03769881', 'minibus'),('n04461696', 'tow_truck'),
 ('n03459775', 'grille'),('n02930766', 'cab'),('n02974003', 'car_wheel'),('n03796401', 'moving_van'),('n03977966', 'police_van'),
 ('n04252225', 'snowplow'),('n02701002', 'ambulance'),('n04065272', 'recreational_vehicle'),('n04467665', 'trailer_truck'),('n03445924', 'golfcart'),
 ('n03345487', 'fire_engine'),('n03791053', 'motor_scooter'),('n03776460', 'mobile_home'),('n04252077', 'snowmobile'),('n02704792', 'amphibian'),
 ('n03417042', 'garbage_truck'),('n02965783', 'car_mirror')]

global graph

graph = tf.get_default_graph()

def prepare_flat(img_224):
    base_model = load_model('static/vgg16.h5')
    model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
    feature = model.predict(img_224)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis=0)
    return flat


CLASS_INDEX_PATH = 'static/imagenet_class_index.json'

def get_predictions(preds, top=5):
    
    global CLASS_INDEX
    CLASS_INDEX = json.load(open(CLASS_INDEX_PATH))

    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
        print(result)
    return results

def car_categories_check(img_224):
    first_check = load_model('static/vgg16.h5')
    print ("Validating that this is a picture of your car...")
    out = first_check.predict(img_224)
    top = get_predictions(out, top=5)
    for j in top[0]:
        if j[0:2] in car_list:
            print ("Car Check Passed!!!")
            print ("\n")
            return True 
    return False

def car_damage_check(img_flat):
    
    second_check = pk.load(open('static/second_check.pickle', 'rb'))
    print ("Validating that damage exists...")
    train_labels = ['Damaged', 'Not Damaged']
    preds = second_check.predict(img_flat)
    prediction = train_labels[preds[0]]

    if prediction == 'Damaged':
        print ("Validation complete - proceeding to location and severity determination")
        print ("\n")
        return True
    else:
        return False

def location_assessment(img_flat):
    print ("Validating the damage area - Front, Rear or Side")
    third_check = pk.load(open("static/third_check.pickle", 'rb'))
    train_labels = ['Front', 'Rear', 'Side']
    preds = third_check.predict(img_flat)
    prediction = train_labels[preds[0]]
    print ("Your Car is damaged at - " + train_labels[preds[0]])
    print ("Location assesment complete")
    print("\n")
    return prediction

def severity_assessment(img_flat):
    print ("Validating the Severity...")
    fourth_check = pk.load(open("static/fourth_check.pickle", 'rb'))
    train_labels = ['Minor', 'Moderate', 'Severe']
    preds = fourth_check.predict(img_flat)
    prediction = train_labels[preds[0]]
    print ("Your Car damage impact is - " + train_labels[preds[0]])
    print ("Severity assesment complete")
    print ("\n")
    print ("Thank you for using the assesment kit from Ashar Siddiqui!!!")
    print ("More such kits in pipeline")
    return prediction


# def location_assessment(img_flat):
#     print ("Validating the damage area - Front, Rear or Side")
#     third_check = pk.load(open('static/third_check.pikle', 'rb'))
#     train_labels = ['Front Damage', 'Rear Damage', 'Side Damage']
#     preds = third_check.predict(img_flat)
#     prediction = train_labels[preds[0]]
#     print ("Your Car is damaged at - " + train_labels[preds[0]])
#     print("Location assesment complete")
#     print("\n")
#     return prediction

# def severity_assessment(img_flat):
#     print("Validate the Severity...")
#     fourth_check = pk.load(open('static/fourth_check.pickle', 'rb'))
#     train_labels = ['Minor Damage', 'Moderate Damage', 'Severe Damage']
    
#     preds = fourth_check.predict(img_flat)
#     prediction = train_labels[preds[0]]
#     print ("Your Car damage impact is - " + train_labels[preds[0]])
#     print ("Severity assesment complete")
#     print ("\n")
#     return prediction

# def engine(request):
    
#     MyCar=request.session['image_path']
#     img_path = MyCar
#     request.session.pop('image_path',None)
#     request.session.modified = True
#     with graph.as_default():

#         img_224 = prepare_img_224(img_path)
#         img_flat = prepare_flat(img_224)
#         g1 = car_categories_check(img_224)
#         g2 = car_damage_check(img_flat)

#         # g3 = ''
#         # g4 = ''
#         # ns = ''

#         while True:
#             try:
#                 if g1 is False:
#                     g1_pic = "Are you sure it is the car"
#                     g2_pic = 'N/A'
#                     g3='N/A'
#                     g4='N/A'
#                     ns='N/A'
#                     break
#                 else:
#                     g1_pic = "it's a car"

#                 if g2 is False:
#                     g2_pic = "are you sere your car is damaged? Make sure you click a clear"
#                     g3='N/A'
#                     g4='N/A'
#                     ns='N/A'
#                     break
#                 else:
#                     g2_pic = "Car Damaged. Refer below sections for locations and Severity"

#                     g3 = location_assessment(img_flat)
#                     g4 = severity_assessment(img_flat)
#                     ns='a). Create a report and send to Vendor \n b). Procced to cost estimate'
#                     break
            
#             except:
#                 break

#     src = 'pic_upload/'
#     import os
#     for image_file_name in os.listdir(src):
#         if image_file_name.endswith(".jpg"):
#             os.remove(src + image_file_name)

#     K.clear_session()

#     return render(
#         request, 'results.html', context={'g1_pic':g1_pic,'g2_pic':g2_pic,'loc':g3,'sev':g4,'ns':ns}
#     )

def engine(request):

    MyCar=request.session['image_path']
    img_path = MyCar
    request.session.pop('image_path', None)
    request.session.modified = True
    with graph.as_default():

        img_224 = prepare_img_224(img_path)
        img_flat = prepare_flat(img_224)
        g1 = car_categories_check(img_224)
        g2 = car_damage_check(img_flat)
        g3 = location_assessment(img_flat)
        g4=severity_assessment(img_flat)

        while True:
            try:

                if g1 is False:
                    g1_pic = "Are you sure its a car?Make sure you click a clear picture of your car and resubmit"
                    g2_pic = 'N/A'
                    g3='N/A'
                    g4='N/A'
                    ns='N/A'
                    break
                else:
                    g1_pic = "Its a Car"

                if g2 is False:
                    g2_pic = "Are you sure your car is damaged?. Make sure you click a clear picture of the damaged portion.Please resubmit the pic"
                    g3='N/A'
                    g4='N/A'
                    ns='N/A'
                    break
                else:
                    g2_pic = "Car Damaged. Refer below sections for Location and Severity"

                    g3 = location_assessment(img_flat)
                    g4=severity_assessment(img_flat)
                    ns='a). Create a report and send to Vendor \n b). Proceed to cost estimation \n c). Estimate TAT'
                    break

            except:
                break


    src= 'pic_upload/'
    import os
    for image_file_name in os.listdir(src):
        if image_file_name.endswith(".jpg") :
            os.remove(src + image_file_name)

    K.clear_session()

    return render(
            request,
            'results.html',context={'g1_pic':g1_pic,'g2_pic':g2_pic, 'loc':g3, 'sev':g4,'ns':ns})