import sys
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import QFile, QPoint, QRect, Qt
from PyQt5.QtWidgets import (QAction, QFileDialog, QMenu, QStyle, QStyleOption, QWidget, QLCDNumber, QSlider, QVBoxLayout, QApplication)
from PoseLabel.ImageView import ImageView



# cooperate_view.setBorder('border:1px solid red')
# cooperate_view.setFocusPolicy(Qt.ClickFocus)  

        

app = QApplication(sys.argv)

cooperate_view = ImageView()
cooperate_view.resize(1280, 720)
cooperate_view.setWindowTitle('Spacial Circle Label')

main_window = cooperate_view
main_window.show()

sys.exit(app.exec_())
