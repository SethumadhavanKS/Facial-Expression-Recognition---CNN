from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
import sys
import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
from keras.models import load_model
from pyqtgraph.Qt import QtCore, QtGui
from  matplotlib.backends.backend_qt5agg  import  ( NavigationToolbar2QT  as  NavigationToolbar )



class StartWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('startWindow.ui', self)
        self.setWindowTitle("Facial Expression Recognition")
        self.button = self.findChild(QtWidgets.QPushButton, 'btn')
        self.button.clicked.connect( self.window2)
        
        self.show()

    # def printButtonPressed(self):
    #     # This is executed when the button is pressed
    #     print('Hello World')
    #     # os.system("python --version")


    def window2(self):                                    
        self.w = MainWindow()
        self.w.show()
        self.hide()
        
class MainWindow(QtWidgets.QMainWindow):                          
    def __init__(self):
        super().__init__()
        self.ls=[]
        self.setWindowTitle("Facial Expression Recognition")
        uic.loadUi('mainWindow.ui', self)


        #Wencam labels
        
        self.webcam=self.findChild(QtWidgets.QPushButton, 'webcam')
        self.label=self.findChild(QtWidgets.QLabel, 'img_lbl')
        self.expr_lbl=self.findChild(QtWidgets.QLabel, 'expr_lbl')
        self.save_btn=self.findChild(QtWidgets.QPushButton, 'save_btn')
        self.close_btn=self.findChild(QtWidgets.QPushButton, 'close_btn')
        self.webc_start=self.findChild(QtWidgets.QPushButton, 'webc_start')
        self.graph_close_btn=self.findChild(QtWidgets.QPushButton, 'graph_close')
        self.webc_frame=self.findChild(QtWidgets.QFrame, 'webc_frame')
        self.video_frame=self.findChild(QtWidgets.QFrame, 'video_frame')
        self . addToolBar ( NavigationToolbar ( self . MplWidget . canvas ,  self ))
        self.timer = QTimer()


        self.graph_close_btn.clicked.connect(self.graph_Close)
        self.webcam.clicked.connect(self.wbcam)
        self.save_btn.clicked.connect(self.save_w)
        self.webc_start.clicked.connect(self.controlTimer)
        self.close_btn.clicked.connect(self.timerClose)
        self.quit_button=self.findChild(QtWidgets.QPushButton, 'quit')
        self.quit_button.clicked.connect(self.closeEvent)   

        # set timer timeout callback function
        self.timer.timeout.connect(self.facedet)

        #Video labels
        self.fileDialog = QtGui.QFileDialog(self)
        self.label_v=self.findChild(QtWidgets.QLabel, 'img_lbl_v')
        self.expr_lbl_v=self.findChild(QtWidgets.QLabel, 'expr_lbl_v')
        self.close_btn_v=self.findChild(QtWidgets.QPushButton, 'close_btn_v')
        self.graph_close_btn_v=self.findChild(QtWidgets.QPushButton, 'graph_close_v')
        self.upload_btn_v=self.findChild(QtWidgets.QPushButton, 'upload_btn_v')
        self.save_btn_v=self.findChild(QtWidgets.QPushButton, 'save_btn_v')
        self.video_btn=self.findChild(QtWidgets.QPushButton, 'video_btn')

        # self.video_frame=self.findChild(QtWidgets.QFrame, 'video_frame')
        self . addToolBar ( NavigationToolbar ( self . MplWidget_v . canvas ,  self ))
        self.timer_v = QTimer()


        self.video_btn.clicked.connect(self.video)
        # self.timer.timeout.connect(self.facedet)
        self.upload_btn_v.clicked.connect(self.video_upload)
        self.close_btn_v.clicked.connect(self.timerClose_v)
        self.timer_v.timeout.connect(self.facedet_v)
        self.graph_close_btn_v.clicked.connect(self.graph_Close_v)
        self.save_btn_v.clicked.connect(self.save_w)


        #Image labels
        self.fileDialog_i = QtGui.QFileDialog(self)
        self.image_frame=self.findChild(QtWidgets.QFrame, 'image_frame')
        self.imgg=[]
        self.label_i=self.findChild(QtWidgets.QLabel, 'img_lbl_i')
        self.expr_lbl_i=self.findChild(QtWidgets.QLabel, 'expr_lbl_i')
        self.close_btn_i=self.findChild(QtWidgets.QPushButton, 'close_btn_i')
        self.upload_btn_i=self.findChild(QtWidgets.QPushButton, 'upload_btn_i')
        self.save_btn_i=self.findChild(QtWidgets.QPushButton, 'save_btn_i')
        self.image_btn=self.findChild(QtWidgets.QPushButton, 'image_button')

        self.image_btn.clicked.connect(self.img)
        self.upload_btn_i.clicked.connect(self.image_upload)
        self.close_btn_i.clicked.connect(self.img_close)
        self.save_btn_i.clicked.connect(self.save_img)

    #main
        self.image_frame.hide()
        self.video_frame.hide()
        self.model = load_model("Model/weights.hdf5", compile=False)
        self.frame=list()
        self.i=0

        


    #Webcam

    def  update_graph ( self ):

        count_angry=self.ls.count("angry")
        count_disgust=self.ls.count("disgust")
        count_fear=self.ls.count("fear")
        count_happy=self.ls.count("happy")
        count_sad=self.ls.count("sad")
        count_surprise=self.ls.count("surprise")
        count_neutral=self.ls.count("neutral")
        #Plot on graph
        self . MplWidget . canvas . axes . clear () 

        y_ax = [count_angry, count_disgust, count_fear, count_happy, count_sad, count_surprise, count_neutral]
        x_ax = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        self . MplWidget . canvas . axes . bar(x_ax,y_ax )
        self . MplWidget . canvas . draw()

        print("Angry:",count_angry,"\n","Disgust:",count_disgust,"\n","Fear:",count_fear,"\n",
        "Happy:",count_happy,"\n","Sad:",count_sad,"\n","Surprise:",count_surprise,"\n","Neutral:",count_neutral)



    def graph_Close(self):
        self . MplWidget . canvas . axes . clear()
        self . MplWidget . canvas . draw()
        self . graph_close_btn . setText('Clear')
  
    def wbcam(self):
        self.webc_frame.show()
        self.image_frame.hide()
        self.video_frame.hide()

    def facedet(self):
        
        
      
        
        face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



    
        ret,test_img=self.cap.read()# captures frame and returns boolean value and captured image
        
        # resize frame image
        scaling_factor = 0.9
        test_img = cv2.resize(test_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    
        if len(faces_detected) == 0:
            self.expr_lbl.setText("")

        for (x,y,w,h) in faces_detected:
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,255,0),thickness=2)
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
            roi_gray=cv2.resize(roi_gray,(64,64))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255



            predictions = self.model.predict(img_pixels)
            # print(predictions)
            #find max indexed array
            max_index = np.argmax(predictions)
            

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            self.ls.append(predicted_emotion)		
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            self.expr_lbl.setText(predicted_emotion)
                   
        self.frame.append(test_img)
        # # convert frame to RGB format
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        # get frame infos
        height, width, channel = test_img.shape
        step = channel * width
        # create QImage from RGB frame
        qImg = QImage(test_img.data, width, height, step, QImage.Format_RGB888)
        # show frame in img_label
        self.label.setPixmap(QPixmap.fromImage(qImg))
        # self.label.setText("hello")


    
    def save_w(self):

        if len(self.frame) >= 1:
            options = self.fileDialog_i.Options()
            # options |= self.fileDialog_i.DontUseNativeDialog
            fileName= self.fileDialog_i.getSaveFileName(self, "Save","untitled.mkv", "Videos (*.mp4 *.mkv)", options=options)
            
            if fileName:
            
                pathOut = fileName[0]
                fps = 10
                frame_array = self.frame
                
                cv2.imwrite('./temp/kan.jpg',self.frame[0])
                img = cv2.imread('./temp/kan.jpg')
                height, width, layers = img.shape
                size = (width,height)
                out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
                for i in range(len(frame_array)):
                    # writing to a image array
                    out.write(frame_array[i])
                out.release() 

    def controlTimer(self):
        self.frame.clear()
        self.ls=[]
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text
            
        # if timer is started
    
    def timerClose(self):
          
            self.timer.stop()
            # release video capture
            self.cap.release()
            self.label.clear()
            self.expr_lbl.setText("")
            self.update_graph()


#Video


    def video(self):
        self.webc_frame.hide()
        self.image_frame.hide()
        self.video_frame.show()
    
    def video_upload(self):
        self.expr_lbl_v.setText("")
        filters = "Videos (*.mkv *.mp4)"
        selected_filter = "Videos (*.mkv *.mp4)"
        fname = self.fileDialog.getOpenFileName(self, " File dialog ", filters, selected_filter)
        # fname = self.fileDialog.getExistingDirectory(self, 'Select directory')
        self.frame.clear()
        imagePath = fname[0]
        if imagePath:
            
            self.ls=[]
            # if timer is stopped
            if not self.timer_v.isActive():
                # create video capture
                self.cap = cv2.VideoCapture(imagePath)
                # start timer
                self.timer_v.start(5)
                # update control_bt text


    def graph_Close_v(self):
        self . MplWidget_v . canvas . axes . clear()
        self . MplWidget_v . canvas . draw()
        # self . graph_close_btn . setText('Clear')
  
        
    
    
    def timerClose_v(self):
            print(len(self.ls))
            self.timer_v.stop()
            # release video capture
            self.cap.release()
            self.label_v.clear()
            self.expr_lbl_v.setText("")
            self.update_graph_v() 

    def facedet_v(self):
       
        
        face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



    
        ret,test_img=self.cap.read()# captures frame and returns boolean value and captured image
        
        # resize frame image
        # scaling_factor = 0.9
        # test_img = cv2.resize(test_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    
        if len(faces_detected) == 0:
            self.expr_lbl.setText("")
        
        for (x,y,w,h) in faces_detected:
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,255,0),thickness=2)
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
            roi_gray=cv2.resize(roi_gray,(64,64))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255



            predictions = self.model.predict(img_pixels)
            #find max indexed array
            max_index = np.argmax(predictions)
            

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            self.ls.append(predicted_emotion)		
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            self.expr_lbl_v.setText(predicted_emotion)
            # cv2.imwrite('./temp/kan'+str(self.i)+'.jpg',test_img)
            # self.i=self.i+1
        
        
        self.frame.append(test_img)    
        # test_img = cv2.resize(test_img, 581, 311)

           
        # convert frame to RGB format
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_img = cv2.resize(test_img, (581, 311))

        # get frame infos
        height, width, channel = test_img.shape
        step = channel * width
        # create QImage from RGB frame
        qImg = QImage(test_img.data, width, height, step, QImage.Format_RGB888)
        # show frame in img_label
        self.label_v.setPixmap(QPixmap.fromImage(qImg))
        # self.label.setText("hello") 
        

    def  update_graph_v ( self ):

        count_angry=self.ls.count("angry")
        count_disgust=self.ls.count("disgust")
        count_fear=self.ls.count("fear")
        count_happy=self.ls.count("happy")
        count_sad=self.ls.count("sad")
        count_surprise=self.ls.count("surprise")
        count_neutral=self.ls.count("neutral")
        #Plot on graph
        self . MplWidget_v . canvas . axes . clear () 

        y_ax = [count_angry, count_disgust, count_fear, count_happy, count_sad, count_surprise, count_neutral]
        x_ax = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        self . MplWidget_v . canvas . axes . bar(x_ax,y_ax )
        self . MplWidget_v . canvas . draw()

        print("Angry:",count_angry,"\n","Disgust:",count_disgust,"\n","Fear:",count_fear,"\n",
        "Happy:",count_happy,"\n","Sad:",count_sad,"\n","Surprise:",count_surprise,"\n","Neutral:",count_neutral)


    #Image
    def img(self):
        self.image_frame.show()
        self.webc_frame.hide()
        self.video_frame.hide()

    def image_upload(self):
        self.expr_lbl_i.setText("")
        filters = "Images (*.jpeg *.png *.jpg)"
        selected_filter = "Images (*.jpeg *.png *.jpg)"
        fname = self.fileDialog_i.getOpenFileName(self, " Select image ", filters, selected_filter)
        # fname = self.fileDialog.getExistingDirectory(self, 'Select directory')
       
        imagePath = fname[0]
        print(imagePath)
        imge = cv2.imread(imagePath)
        self.facedet_i(imge)

    def facedet_i(self,imge):

        
        #load model
        # model = model_from_json(open("Model/fer.json", "r").read())
        # #load weights
        # model.load_weights('Model/fer.h5')
        
        face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



        gray_img= cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    
        for (x,y,w,h) in faces_detected:
            cv2.rectangle(imge,(x,y),(x+w,y+h),(255,255,0),thickness=2)
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
            roi_gray=cv2.resize(roi_gray,(64,64))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255

            predictions = self.model.predict(img_pixels)

            #find max indexed array
            max_index = np.argmax(predictions[0])
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            self.expr_lbl_i.setText(predicted_emotion)
        
            cv2.putText(imge, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        self.imgg=imge
        height, width, channel = imge.shape
        ratio=width/height
        w=311*ratio
        imge = cv2.resize(imge, (int(w), 311))

        # convert frame to RGB format
        test_img = cv2.cvtColor(imge, cv2.COLOR_BGR2RGB)
        # scaling_factor = 0.9
        # test_img = cv2.resize(test_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        # self.imgg=imge
        # get frame infos
        height, width, channel = test_img.shape
        step = channel * width
        # create QImage from RGB frame
        qImg = QImage(test_img.data, width, height, step, QImage.Format_RGB888)
        # show frame in img_label
        self.label_i.setPixmap(QPixmap.fromImage(qImg))
    

    def img_close(self):
        self.label_i.clear()
        self.expr_lbl_i.setText("")

    def save_img(self):
        options = self.fileDialog_i.Options()
        # options |= self.fileDialog_i.DontUseNativeDialog
        fileName= self.fileDialog_i.getSaveFileName(self, "Save","untitled.jpg", "Images (*.jpeg *.png *.jpg)", options=options)
        
        if fileName:
            cv2.imwrite(fileName[0], self.imgg)
            # print("image saved")


    def closeEvent(self, event):
        """Generate 'question' dialog on clicking 'X' button in title bar.

        Reimplement the closeEvent() event handler to include a 'Question'
        dialog with options on how to proceed - Save, Close, Cancel buttons
        """
        reply = QMessageBox.question(
            self, "Message",
            "Are you sure you want to quit? ",
             QMessageBox.Close | QMessageBox.Cancel)

        if reply == QMessageBox.Close:
            app.quit()
        else:
            pass


app = QtWidgets.QApplication(sys.argv)
window = StartWindow()
app.exec_()