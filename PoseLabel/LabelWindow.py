from PyQt5 import QtGui
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAction, QFileDialog, QLabel, QMenu, QStyle, QStyleOption, QWidget)
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget
from ImageView import ImageView


class LabelWindow(QMainWindow):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        
        self.resize(1600, 900)
        self.setWindowTitle('Spacial Circle Label')
        self.center()
                
        
    
        cooperate_view = ImageView(self)
        cooperate_view.resize(750, 600)
        cooperate_view.move(20, 200)
        cooperate_view.setBorder('border:1px solid red')
        cooperate_view.setFocusPolicy(Qt.ClickFocus)
        
        
        
        
        
    def center(self):
        screen = QDesktopWidget().screenGeometry() #计算显示屏的尺寸
        size = self.geometry()          #计算qwidget的窗口大小
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)  #把窗口移动到屏幕正中央
        
        
    # def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        
    #     print(a0.key())
    #     return super().keyPressEvent(a0)
        

        