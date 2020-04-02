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
from pyqtgraph.Qt import QtCore, QtGui



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
        #load model
        self.model = model_from_json(open("Model/fer.json", "r").read())
        #load weights
        self.model.load_weights('Model/fer.h5')
        self.frame=list()
        self.i=0

    #Webcam



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