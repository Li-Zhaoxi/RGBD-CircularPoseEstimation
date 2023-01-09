import json
import os
from PyQt5 import QtGui
from PyQt5.QtCore import QFile, QLineF, QPoint, QPointF, QRect, QRectF, Qt
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel, QMenu, QMessageBox, QStyle, QStyleOption, QWidget)

import numpy as np
from ElpPy.utils import ElliFit


from PoseLabel.Shapes import Shapes


# 补充点的标记，线的标记
# 开启Pose标记之后：打开椭圆标记表示开始标记椭圆
# 按下s同样表示一个图形的标记成功，只不过按下回车则表示一个Pose的标记成功
# 回车后：清空已标记的椭圆和其他图形

# 精准标记点和线，存在反变换的问题




class ImageView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        
        self.m_Image = QtGui.QImage()
        self.m_ZoomValue = 1.0
        self.m_XPtInterval = 0
        self.m_YPtInterval = 0
        self.m_OldPos = QPoint 
        self.m_Pressed = False
        
        
        
        self.labeled_shapes = Shapes()
        
        self.image_full_path = ''
        self.image_full_paths = []
        self.current_image_index = -1
        
        self.m_StartLabelEllipse = False
        self.m_StartLabelCircle = False
        self.m_StartLabelPoint = False
        self.m_StartLabelLine = False

        
        
    def setBorder(self, stype_sheet:str):
        label_view = QLabel(parent=self)
        label_view.resize(self.width(), self.height())
        label_view.setStyleSheet(stype_sheet)
        
    def contextMenuEvent(self, a0: QtGui.QContextMenuEvent) -> None:
        pos = a0.pos()
        pos = self.mapToGlobal(pos)
        menu = QMenu(self)
        
        loadImage = QAction("Load Image")
        loadImage.triggered.connect(self.onLoadImage)
        menu.addAction(loadImage)
        
        loadImages = QAction("Load Images")
        loadImages.triggered.connect(self.onLoadImages)
        menu.addAction(loadImages)
        
        loadSPImages = QAction("Load Specific Images")
        loadSPImages.triggered.connect(self.onLoadSpecificImages)
        menu.addAction(loadSPImages)
        
        menu.addSeparator()
        
        zoomInAction = QAction("Zoom In")
        zoomInAction.triggered.connect(self.onZoomInImage)
        menu.addAction(zoomInAction)
        
        zoomOutAction = QAction("Zoom Out")
        zoomOutAction.triggered.connect(self.onZoomOutImage)
        menu.addAction(zoomOutAction)
        
        presetAction = QAction("Preset")
        presetAction.triggered.connect(self.onPresetImage)
        menu.addAction(presetAction)
        
        menu.addSeparator()
        labelelpAction = QAction("Label Ellipse")
        labelelpAction.setCheckable(True)
        labelelpAction.setChecked(self.m_StartLabelEllipse)
        labelelpAction.triggered.connect(self.onLabelEllipse)
        menu.addAction(labelelpAction)
        
        labelcirAction = QAction("Label Circle")
        labelcirAction.setCheckable(True)
        labelcirAction.setChecked(self.m_StartLabelCircle)
        labelcirAction.triggered.connect(self.onLabelCircle)
        menu.addAction(labelcirAction)
        
        labelptAction = QAction("Label Points")
        labelptAction.setCheckable(True)
        labelptAction.setChecked(self.m_StartLabelPoint)
        labelptAction.triggered.connect(self.onLabelPoint)
        menu.addAction(labelptAction)
        
        labelLineAction = QAction("Label Lines")
        labelLineAction.setCheckable(True)
        labelLineAction.setChecked(self.m_StartLabelLine)
        labelLineAction.triggered.connect(self.onLabelLine)
        menu.addAction(labelLineAction)
        
        menu.addSeparator()
        menu_spacial_circle_pose = QMenu(menu)
        menu_spacial_circle_pose.setTitle('Label Circular Pose')
        
        cam_intric_action = QAction("Load ")
        
        simpleLabelAction = QAction("Simple Label")   # 简单标记，不包含逆透视变换
        # clearAllAction.triggered.connect(self.onClearShapes)
        menu_spacial_circle_pose.addAction(simpleLabelAction)
        
        preciseLabelAction = QAction('Precise Label')  # 精确标记，使用逆透视变换，需要相机内参，利用TAB键切换逆透视变换使用的法向
        menu_spacial_circle_pose.addAction(preciseLabelAction)
        
        menu.addMenu(menu_spacial_circle_pose)
        
        menu.addSeparator()
        loadJsonAction = QAction("Load Shapes")
        loadJsonAction.triggered.connect(self.onLoadShapes)
        menu.addAction(loadJsonAction)
        
        saveAction = QAction("Save Shapes")
        saveAction.triggered.connect(self.onSaveShapes)
        menu.addAction(saveAction)
        
        clearAllAction = QAction("Clear Shapes")
        clearAllAction.triggered.connect(self.onClearShapes)
        menu.addAction(clearAllAction)
        
        
        
        print(a0.pos())
        menu.exec(pos)
        
        # return super().contextMenuEvent(a0)
    
    def wheelEvent(self, a0: QtGui.QWheelEvent) -> None:
        wheel_value = a0.angleDelta()
        value = wheel_value.y()
        if value > 0:
            self.onZoomInImage()
        else:
            self.onZoomOutImage()
        
        self.update()
        # return super().wheelEvent(a0)
        
    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QtGui.QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, painter, self)
        
        if self.m_Image.isNull():
            return super().paintEvent(a0)
        
        
        # 计算最终显示在widgets里面的图像的最大分辨率
        width = min(self.m_Image.width(), self.width())
        height = width * 1.0 / (self.m_Image.width() * 1.0 / self.m_Image.height())
        
        height = min(height, self.height())
        width = height * 1.0 * (self.m_Image.width() * 1.0 / self.m_Image.height())
        
        s2 = width * 1.0 / self.m_Image.width()
        
        painter.translate(self.width() / 2 + self.m_XPtInterval, self.height() / 2 + self.m_YPtInterval)
        painter.scale(self.m_ZoomValue, self.m_ZoomValue)
        picRect = QRect(-width/2, -height/2, width, height)
        
        # print(height, width)
        painter.drawImage(picRect, self.m_Image)
        
        
        if self.m_StartLabelEllipse is True or self.m_StartLabelCircle is True:
            self.labeled_shapes.drawSelectElp(painter, QtGui.QPen(Qt.blue), QtGui.QPen(Qt.red),
                                              self.m_ZoomValue, width, height, s2)            
        
        if self.m_StartLabelPoint is True:
            self.labeled_shapes.drawSelectPoint(painter, QtGui.QPen(Qt.blue), self.m_ZoomValue, 
                                                width, height, s2)
        
        if self.m_StartLabelLine is True:
            self.labeled_shapes.drawSelectLine(painter, QtGui.QPen(Qt.blue), QtGui.QPen(Qt.red),
                                              self.m_ZoomValue, width, height, s2)  
        
        self.labeled_shapes.drawLabelElps(painter, QtGui.QPen(Qt.red),QtGui.QPen(Qt.green),
                                          self.m_ZoomValue, width, height, s2)
        self.labeled_shapes.drawLabeledPoints(painter, QtGui.QPen(Qt.red), self.m_ZoomValue, 
                                              width, height, s2)
        self.labeled_shapes.drawLabelLines(painter, QtGui.QPen(Qt.red),QtGui.QPen(Qt.green),
                                          self.m_ZoomValue, width, height, s2)

    # qwidgets上的点坐标转换为图像坐标
    def pos2uv(self, pos:QPoint)->QPointF:
        
        assert(not self.m_Image.isNull())
        
        # 计算最终显示在widgets里面的图像的最大分辨率
        width = min(self.m_Image.width(), self.width())
        height = width * 1.0 / (self.m_Image.width() * 1.0 / self.m_Image.height())
        
        height = min(height, self.height())
        width = height * 1.0 * (self.m_Image.width() * 1.0 / self.m_Image.height())
        
        x1 = pos.x()
        y1 = pos.y()
        
        s1 = self.m_ZoomValue
        cx1 = self.width() / 2 + self.m_XPtInterval
        cy1 = self.height() / 2 + self.m_YPtInterval
        
        s2 = width * 1.0 / self.m_Image.width()
        cx2 = 0
        cy2 = 0
        
        x2 = (x1 - cx1) / s1
        y2 = (y1 - cy1) / s1
        
        # s2 = 1
        u = (x2 + width / 2) / s2
        v = (y2 + height / 2) / s2
        
        print('x1:{0}, y1:{1}, s1:{2}, cx1:{3}, cy1:{4}, x2:{5}, y2:{6}'.format(
            x1, y1, s1, cx1, cy1, x2, y2))
        
        print('x2:{0}, y2:{1}, s2:{2}, cx2:{3}, cy2:{4}, u:{5}, v:{6}'.format(
            x2, y2, s2, cx2, cy2, u, v))
        
        return QPointF(u, v)
    
    def mouseDoubleClickEvent(self, a0: QtGui.QMouseEvent) -> None:
        
        if self.m_Image.isNull():
            return super().mouseDoubleClickEvent(a0)
        
        # 相对左上角的位置
        pos = a0.pos()
        pt = self.pos2uv(pos)
        
        # print('mouseDoubleClickEvent', pos)
        # print('Pixel Position', pt, self.m_Image.size())
        
        if self.m_StartLabelEllipse:
            self.labeled_shapes.add_selected_elp_pt(pt)
        
        if self.m_StartLabelCircle:
            self.labeled_shapes.add_selected_elp_pt(pt, fit_circle=True)
            
        if self.m_StartLabelPoint:
            self.labeled_shapes.add_selected_point_pt(pt)
            
        if self.m_StartLabelLine:
            self.labeled_shapes.add_selected_line_pts(pt)
        
            
        self.update()
        
    
    
    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.m_OldPos = a0.pos()
        self.m_Pressed = True
        # return super().mousePressEvent(a0)
    
    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        if not self.m_Pressed:
            return super().mouseMoveEvent(a0)
        
        self.setCursor(Qt.SizeAllCursor)
        pos = a0.pos()
        xPtInterval = pos.x() - self.m_OldPos.x()
        yPtInterval = pos.y() - self.m_OldPos.y()
        
        self.m_XPtInterval += xPtInterval
        self.m_YPtInterval += yPtInterval
        self.m_OldPos = pos
        self.update()
    
    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.m_Pressed = False
        self.setCursor(Qt.ArrowCursor)

        # return super().mouseReleaseEvent(a0)
    
    def loadImage(self, img_path, autoloadjson = False):
        self.image_full_path = img_path
        file = QFile(self.image_full_path)
        
        if not file.exists():
            print('cannot open file: {0}'.format(self.image_full_path))
            return
        
        # self.current_image_index = min(len(self.image_full_paths) - 1, self.current_image_index + 1)
        self.m_Image.load(self.image_full_path)
        self.labeled_shapes.clearAll()
        self.setWindowTitle('{0}, {1}/{2}'.format(self.image_full_path, self.current_image_index + 1, len(self.image_full_paths)))
        
        label_json_path = self.image_full_path + '.json'
        if os.path.exists(label_json_path):
            if autoloadjson:
                self.loadJson(label_json_path)
            else:
                click = QMessageBox.question(self, "Json Loading", 'whether load the json {0}'.format(label_json_path),  
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if click == QMessageBox.Yes:
                    self.loadJson(label_json_path)
    
    def onLoadSpecificImages(self):
        folder_name = QFileDialog.getExistingDirectory(self, "Open Image Folder", "./")
        if len(folder_name) == 0:
            return 
        
        name_list_path = QFileDialog.getOpenFileName(self, "Open Name List", folder_name, "List (*.txt)")
        name_list_path = name_list_path[0]
        if len(name_list_path) == 0:
            return
        
        img_names = [] # 图像名
        with open(name_list_path, 'r') as f:
            for each_line in f.readlines():
                if len(each_line) < 3:
                    continue
                tmp_name = each_line.strip('\n')
                img_names.append(os.path.join(folder_name, tmp_name))
        
        self.image_full_paths = img_names
        self.current_image_index = 0
        self.loadImage(self.image_full_paths[self.current_image_index], autoloadjson=True)
        self.update()
        
    
    def onLoadImages(self):
        imageFiles = QFileDialog.getOpenFileNames(self, "Open Image", "./", "Images (*.png *.xpm *.jpg)")
        image_names = imageFiles[0]
        # print(imageFiles)
        if len(image_names) == 0:
            return
        self.image_full_paths = image_names
        
        self.current_image_index = 0
        
        self.loadImage(self.image_full_paths[self.current_image_index], autoloadjson=True)
        
        self.update()
        
    
    def onLoadImage(self):
        imageFile = QFileDialog.getOpenFileName(self, "Open Image", "./", "Images (*.png *.xpm *.jpg)")
        image_full_path = imageFile[0]
        
        if len(image_full_path) == 0:
            return
        
        self.image_full_paths = [image_full_path]
        self.current_image_index = 0
        
        self.image_full_path = image_full_path
        
        print(self.image_full_path)
        # print(imageFile[0])
    
        file = QFile(self.image_full_path)
        
        if not file.exists():
            print('cannot open file: {0}'.format(self.image_full_path))
            return
        self.m_Image.load(self.image_full_path)
        self.labeled_shapes.clearAll()
        
        label_json_path = self.image_full_path + '.json'
        if os.path.exists(label_json_path):
            click = QMessageBox.question(self, "Json Loading", 'whether load the json {0}'.format(label_json_path),  
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if click == QMessageBox.Yes:
                self.loadJson(label_json_path)
        
        self.update()
    
    def onZoomInImage(self):
        self.m_ZoomValue += 0.2
        self.update()
    
    def onZoomOutImage(self):
        self.m_ZoomValue -= 0.2
        if self.m_ZoomValue <= 0:
            self.m_ZoomValue += 0.2
            return
        
        self.update()
    
    def onPresetImage(self):
        self.m_ZoomValue = 1.0
        self.m_XPtInterval = 0
        self.m_YPtInterval = 0
        self.update()
        
    def onLabelCircle(self):
        if self.m_StartLabelCircle is True:
            self.m_StartLabelCircle = False
            self.labeled_shapes.clearSelectElp()
        else:
            if not self.m_Image.isNull():
                self.labeled_shapes.clearSelectElp()
                self.m_StartLabelCircle = True
            else:
                QMessageBox.critical(self, "Empty Image", "You haven't loaded the picture yet")
        self.update()
    
    def onLabelEllipse(self):
        if self.m_StartLabelEllipse is True:
            self.m_StartLabelEllipse = False
            # self.labeled_shapes.clearLabeledElps() 
            self.labeled_shapes.clearSelectElp() # 不在标记的时候清空标记点
        else:
            if not self.m_Image.isNull():
                # self.labeled_shapes.clearLabeledElps() 
                self.labeled_shapes.clearSelectElp() # 使用前清空标记点
                self.m_StartLabelEllipse = True
            else:
                QMessageBox.critical(self, "Empty Image", "You haven't loaded the picture yet")
        self.update()
    
    def onLabelLine(self):
        if self.m_StartLabelLine is True:
            self.m_StartLabelLine = False
            self.labeled_shapes.clearSelectLine()
        else:
            if not self.m_Image.isNull():
                self.labeled_shapes.clearSelectLine()
                self.m_StartLabelLine = True
            else:
                QMessageBox.critical(self, "Empty Image", "You haven't loaded the picture yet")
    
    def onLabelPoint(self):
        if self.m_StartLabelPoint is True:
            self.m_StartLabelPoint = False
            self.labeled_shapes.clearSelectPoint()
        else:
            if not self.m_Image.isNull():
                self.labeled_shapes.clearSelectPoint()
                self.m_StartLabelPoint = True
            else:
                QMessageBox.critical(self, "Empty Image", "You haven't loaded the picture yet")
        self.update()
    
    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        
        usage_key = a0.key()
        # print(usage_key, Qt.Key_S)
        if usage_key == Qt.Key_S:
            
            if QApplication.keyboardModifiers() == Qt.ControlModifier: # ctrl + S
                self.onSaveShapes()
            else: # S only
                if self.m_StartLabelEllipse or self.m_StartLabelCircle: # 椭圆标记模式
                    self.labeled_shapes.add_selected_elp()
                
                if self.m_StartLabelPoint:
                    self.labeled_shapes.add_selected_point()
                    
                if self.m_StartLabelLine:
                    self.labeled_shapes.add_selected_line()
                    
            
            self.update()
            return
        
        if usage_key == Qt.Key_Escape: # ESC
            
            if self.m_StartLabelEllipse or self.m_StartLabelCircle: 
                self.labeled_shapes.clearSelectElp()
                
            if self.m_StartLabelPoint:
                self.labeled_shapes.clearSelectPoint()
                
            if self.m_StartLabelLine:
                self.labeled_shapes.clearSelectLine()
                
            self.update()
            return
        
        if usage_key == Qt.Key_Up:
            if self.labeled_shapes.checkChanged():
                QMessageBox.critical(self, "Label Changed", "You haven't save the modified shapes")
            else:
                self.current_image_index = max(0, self.current_image_index - 1)
                # print('current_image_index', self.current_image_index)
                self.loadImage(self.image_full_paths[self.current_image_index], autoloadjson=True)
            
            self.update()
            return
        
        if usage_key == Qt.Key_Down:
            if self.labeled_shapes.checkChanged():
                QMessageBox.critical(self, "Label Changed", "You haven't save the modified shapes")
            else:
                self.current_image_index = min(len(self.image_full_paths) - 1, self.current_image_index + 1)
                # print('current_image_index', self.current_image_index)
                self.loadImage(self.image_full_paths[self.current_image_index], autoloadjson=True)
            
            self.update()
            return
                
        
        return super().keyPressEvent(a0)
        
    def onSaveShapes(self):
        if self.m_Image.isNull():
            return 
        json_full_path = self.image_full_path + '.json'
        self.labeled_shapes.saveJson(json_full_path)
        
        QMessageBox.information(self, "Save Labeled Shapes", 'saved in {0}'.format(json_full_path))
        
        # QMessageBox::information(NULL, "Title", "Content");
    
    def loadJson(self, json_path):
        self.labeled_shapes = Shapes()
        self.labeled_shapes.loadJson(json_path)
    
    def onLoadShapes(self):
        if self.m_Image.isNull():
            return 
        
        json_file = QFileDialog.getOpenFileName(self, "Open Json File", "./", "Json (*.json)")
        json_file = json_file[0]
        print(json_file)
        if len(json_file) == 0:
            return
        
        self.loadJson(json_file)
        
        self.update()
        

    def onClearShapes(self):
        self.labeled_shapes.clearAll()
        self.update()