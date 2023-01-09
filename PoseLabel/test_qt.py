import sys
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import QFile, QPoint, QRect, Qt
from PyQt5.QtWidgets import (QAction, QFileDialog, QMenu, QStyle, QStyleOption, QWidget, QLCDNumber, QSlider, QVBoxLayout, QApplication)
from LabelWindow import LabelWindow
class SigSlot(QWidget):
    def __init__(self,parent=None):
        QWidget.__init__(self)
        self.setWindowTitle('XXOO')
        lcd = QLCDNumber(self)
        slider = QSlider(Qt.Horizontal,self)
         
        vbox = QVBoxLayout()
        vbox.addWidget(lcd)
        vbox.addWidget(slider)
         
        self.setLayout(vbox)
        
         
        slider.valueChanged.connect(lcd.display)
        self.resize(350,250)
        
        
   
# app = QApplication(sys.argv)
# qb = SigSlot()


# qb.show()
# sys.exit(app.exec_())

app = QApplication(sys.argv)

main_window = LabelWindow()
main_window.show()
sys.exit(app.exec_())
