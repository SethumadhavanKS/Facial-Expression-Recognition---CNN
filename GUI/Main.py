from PyQt5 import QtWidgets, uic
import sys
import os


class Window2(QtWidgets.QMainWindow):                          
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Window22222")
        uic.loadUi('main2.ui', self)

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('direct.ui', self)

        self.button = self.findChild(QtWidgets.QPushButton, 'btn')
        self.button.clicked.connect( self.window2)

        self.show()

    # def printButtonPressed(self):
    #     # This is executed when the button is pressed
    #     print('Hello World')
    #     # os.system("python --version")
######

    def window2(self): 
        print("hello")                                    
        self.w = Window2()
        self.w.show()
        self.hide()

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()