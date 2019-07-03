import cv2
import numpy as np
import os
import torch
from keras.preprocessing import image
from gan_celeb import Generator, Discriminator

gen_model = torch.load('model/generator.pt')
dis_model = torch.load('model/discriminator.pt')

path = os.getcwd()
face_cascade = cv2.CascadeClassifier('haarcascades/frontal_face_default.xml') 

output_folder = path + '/Output/'

input_folder = path + '/testing/'
filename = '1.jpg'

foldername = output_folder + str(filename.split('.')[0]) + '/'
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
if not os.path.exists(foldername):
    os.makedirs(foldername)

img = cv2.imread(input_folder + filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detects faces of different sizes in the input image 
faces = face_cascade.detectMultiScale(gray, 1.2, 4) 

dis_model.zero_grad()
fixed_noise = torch.randn(1, 100, 1, 1, device=device)

if len(faces) >=1:
    print('Face found')
    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
    #    cv2.rectangle(img,(x-20,y-20),(x+w+20,y+h+20),(255,255,0),2)  
        roi_gray = gray[y-20:y+h+20, x-20:x+w+20] 
        roi_color = img[y-20:y+h+20, x-20:x+w+20] 
    
    gray_image = cv2.resize(roi_color, (64,64))
    gray_image = gray_image.reshape([3,64,64])
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x = image.img_to_array(gray_image)
    x = np.expand_dims(x, axis = 0)
    
    x /= 255
    
    x = torch.tensor(x).to(device)
#    batch = x.size(0)
#    label = torch.full((batch,), 1, device=device)
#    
#    output = dis_model(x).view(-1)
#    noise = torch.randn(batch, 100, 1, 1, device=device)
#    
#    fake = gen_model(noise)
#    label.fill_(0)
#    
#    output = dis_model(fake.detach()).view(-1)
#    
#    output = dis_model(fake).view(-1)
#    
#    with torch.no_grad():
#        fake = gen_model(fixed_noise).detach().cpu()
    fake = fake.reshape([64,64,3])
    fake = fake.cpu().detach().numpy()
    fake = cv2.resize(fake, (img.shape[:2]))
    
    cv2.imwrite('Original_face.png', roi_color)
    cv2.imwrite('Fake_face.png', fake*255)
    
else:
    print('Face not detected')