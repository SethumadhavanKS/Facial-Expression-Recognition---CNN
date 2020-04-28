import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import pandas as pd

import matplotlib.pyplot as plt

        img_pixels /= 255

        predictions = model.predict(img_pixels)
        #find max indexed array
        max_index = np.argmax(predictions)
        

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        ls.append(predicted_emotion)		
	    
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break
print(ls)
count_angry=ls.count("angry")
count_disgust=ls.count("disgust")
count_fear=ls.count("fear")
count_happy=ls.count("happy")
count_sad=ls.count("sad")
count_surprise=ls.count("surprise")
count_neutral=ls.count("neutral")
#Plot on graph
y_ax = [count_angry, count_disgust, count_fear, count_happy, count_sad, count_surprise, count_neutral]
x_ax = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
barlist=plt.bar( x_ax, y_ax)
barlist[0].set_color('r')
barlist[1].set_color('y')
barlist[3].set_color('g')
barlist[4].set_color('c')
barlist[5].set_color('k')
barlist[6].set_color('m')
plt.show()



print("Angry:",count_angry,"\n","Disgust:",count_disgust,"\n","Fear:",count_fear,"\n",
"Happy:",count_happy,"\n","Sad:",count_sad,"\n","Surprise:",count_surprise,"\n","Neutral:",count_neutral)
cap.release()
cv2.destroyAllWindows
#