import cv2
from PoseLabel.label_utils import cvtpts2Qptfs, cvtQPtfs2pts
from PyQt5 import QtGui
from PyQt5.QtCore import QLineF, QPointF, QRectF
from ElpPy.utils import ElliFit, GeneralEllipse, GeneralLine
import numpy as np
from json import JSONEncoder
import json


# 补充序列化，反序列化
# 再shapes里面补充相关数据

class ShapesEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, GeneralEllipse):
            return o.getDict()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, QPointF):
            return [o.x(), o.y()]
        if isinstance(o, GeneralLine):
            return o.getDict()
        else:
            return super().default(o)





# 记录一些初级数据啥的
class Shapes(object):
    def __init__(self) -> None:
        super().__init__()
        
        # 标记椭圆需要的一些数据
        self.selected_elp_pts = []
        self.selected_elp_shape = None
        self.labeled_elps = []
        self.name_ellipse = 'ellipse'
        
        # 标记点需要的一些数据
        self.selected_point = None
        self.labeled_pts = []
        self.name_point = 'point'
        
        # 标记直线需要的一些数据
        self.selected_line_pts = []
        self.selected_line_shape = None
        self.labeled_lines = []
        self.name_line = 'line'
        
        # 标记是否数据改变
        self.have_changed = False
        
        
    def clearAll(self):
        self.clearSelectElp()
        self.clearLabeledElps()
        
        self.clearSelectPoint()
        self.clearLabeledPoint()
        
        self.clearSelectLine()
        self.clearLabeldLine()
    
    def drawAllOnImage(self, imgC, 
                       color_ellipse, color_elppts, color_selpts,
                       color_line, color_linepts,
                       thikness = 2, offset = None):
        self.drawLabelElpsOnImage(imgC, color_ellipse, color_elppts, thikness, offset)
        self.drawLabelPointsOnImage(imgC, color_selpts, offset)
        self.drawLabelLinesOnImage(imgC, color_line, color_linepts, thikness, offset)
        
        
        
    def saveJson(self, save_path):
        
        save_dict = {}
        
        if len(self.labeled_elps) > 0:
            save_dict[self.name_ellipse] = self.labeled_elps
            
        if len(self.labeled_pts) > 0:
            save_dict[self.name_point] = self.labeled_pts
        
        if len(self.labeled_lines) > 0:
            save_dict[self.name_line] = self.labeled_lines
        
        json_str = json.dumps(save_dict, indent=1, cls=ShapesEncoder)
        
        with open(save_path, 'w') as f:
            f.write(json_str)
            f.close()
            
        self.have_changed = False
    
    def loadJson(self, json_path):
        
        with open(json_path, 'r') as load_f:
            load_dict = json.load(load_f)
        
        shape_keys = list(load_dict.keys())
        
        if self.name_ellipse in shape_keys:
            self.loadEllipseData(load_dict[self.name_ellipse])
            
        if self.name_point in shape_keys:
            self.loadPointsData(load_dict[self.name_point])
            
        if self.name_line in shape_keys:
            self.loadLinesData(load_dict[self.name_line])
            
        self.have_changed = False
    
    # 检查数据相对于最开始是否有改变
    def checkChanged(self):
        return self.have_changed
    
    def loadLinesData(self, line_dict):
        num_labeled_lines = len(line_dict)
        self.clearSelectLine()
        self.labeled_lines = [[] for _ in range(num_labeled_lines)]
        for idx_labeled_line in range(num_labeled_lines):
            each_label_line = {}
            gline = GeneralLine()
            gline.loadData(line_dict[idx_labeled_line]['gline'])
            each_label_line['gline'] = gline
                        
            pts_np = line_dict[idx_labeled_line]['pts']
            pts_qt = cvtpts2Qptfs(pts_np)
            each_label_line['pts'] = pts_qt
            
            self.labeled_lines[idx_labeled_line] = each_label_line      
    
    def loadPointsData(self, point_dict): 
        self.clearSelectPoint()
        ptsnp = np.array(point_dict)
        ptsqt = cvtpts2Qptfs(ptsnp)
        self.labeled_pts = ptsqt
    
    def loadEllipseData(self, ellipse_dict):
        num_labeled_elp = len(ellipse_dict)
        self.clearSelectElp()        
        self.labeled_elps = [[] for _ in range(num_labeled_elp)]
        for idx_labeled_elp in range(num_labeled_elp):
            each_label_elp = {}
            pts_np = ellipse_dict[idx_labeled_elp]['pts']
            pts_qt = cvtpts2Qptfs(pts_np)
            each_label_elp['pts'] = pts_qt
            each_label_elp['gelp'] = GeneralEllipse(gelp_dict=ellipse_dict[idx_labeled_elp]['gelp'])
            self.labeled_elps[idx_labeled_elp] = each_label_elp
        
    ################### Line ###############################
    def add_selected_line_pts(self, pt):
        self.selected_line_pts.append(pt)
        if len(self.selected_line_pts) >= 2:
            pts = cvtQPtfs2pts(self.selected_line_pts)
            self.selected_line_shape = GeneralLine(pts = pts)
    
    def add_selected_line(self):
        if self.selected_line_shape is not None:
            self.labeled_lines.append({'pts': self.selected_line_pts,
                                      'gline': self.selected_line_shape})
            self.clearSelectLine()
            self.have_changed = True
    
    def clearSelectLine(self):
        self.selected_line_pts = []
        self.selected_line_shape = None
    
    def clearLabeldLine(self):
        self.labeled_lines = []
                
    def drawSelectLine(self, painter: QtGui.QPainter, pen_pt, pen_line, m_ZoomValue, width, height, imgscale):
        if len(self.selected_line_pts) > 0:
            painter.setPen(pen_pt)
            for each_pt in self.selected_line_pts:     
                r = 3 / m_ZoomValue
                cx = each_pt.x() * imgscale - width/2 - r
                cy = each_pt.y() * imgscale - height/2 - r
                draw_pt = QRectF(cx, cy, 2 * r, 2 * r)
                painter.drawEllipse(draw_pt)
        
        if self.selected_line_shape is not None:
            gline = self.selected_line_shape
            x, y = gline.GenerateLineData()
            pts = np.vstack([x,y]).transpose()
            qptfs = cvtpts2Qptfs(pts)
            for p1, p2 in zip(qptfs[:-1], qptfs[1:]):
                painter.setPen(pen_line)
                dp1 = QPointF(p1.x() * imgscale - width/2, p1.y() * imgscale - height/2)
                dp2 = QPointF(p2.x() * imgscale - width/2, p2.y() * imgscale - height/2)
                linef = QLineF(dp1, dp2)
                painter.drawLine(linef)
                
                
    def drawLabelLinesOnImage(self, imgC, color_line, color_points, thickness = 2, offset = None):
        if len(self.labeled_lines) > 0:
            if offset is None:
                offset = np.array([0,0])
            for each_labeled_line in self.labeled_lines:
                gline = each_labeled_line['gline']
                pts = each_labeled_line['pts']
                
                x, y = gline.GenerateLineData()
                line_pts = np.vstack([x + offset[0], y + offset[1]]).transpose()
                
                for p1, p2 in zip(line_pts[:-1,:], line_pts[1:,:]):
                    usage_p1 = np.round(p1).astype(np.int)
                    usage_p2 = np.round(p2).astype(np.int)
                    cv2.line(imgC, (usage_p1[0], usage_p1[1]), 
                             (usage_p2[0], usage_p2[1]), color_line, thickness)
                
                if len(pts) > 0:
                    for each_pt in pts:
                        cx = int(each_pt.x() + 0.5 + offset[0])
                        cy = int(each_pt.y() + 0.5 + offset[1])
                        cv2.circle(imgC, (cx, cy), 2, color_points, 2)
                
    def drawLabelLines(self, painter:QtGui.QPainter, pen_pt, pen_line, m_ZoomValue, width, height, imgscale):
        if len(self.labeled_lines) > 0:
            print(self.labeled_lines)
            for each_labeled_line in self.labeled_lines:
                gline = each_labeled_line['gline']
                pts = each_labeled_line['pts']
                
                x, y = gline.GenerateLineData()
                line_pts = np.vstack([x,y]).transpose()
                qptfs = cvtpts2Qptfs(line_pts)
                for p1, p2 in zip(qptfs[:-1], qptfs[1:]):
                    painter.setPen(pen_line)
                    dp1 = QPointF(p1.x() * imgscale - width/2, p1.y() * imgscale - height/2)
                    dp2 = QPointF(p2.x() * imgscale - width/2, p2.y() * imgscale - height/2)
                    linef = QLineF(dp1, dp2)
                    painter.drawLine(linef)
                    
                if len(pts) > 0:
                    painter.setPen(pen_pt)
                    for each_pt in pts:
                        r = 3 / m_ZoomValue
                        cx = each_pt.x() * imgscale - width/2 - r
                        cy = each_pt.y() * imgscale - height/2 - r
                        draw_pt = QRectF(cx, cy, 2 * r, 2 * r)
                        painter.drawEllipse(draw_pt)
        
    ####################### Point #############################
    
    def add_selected_point_pt(self, pt):
        self.selected_point = pt
    
    def add_selected_point(self):
        self.labeled_pts.append(self.selected_point)
        self.selected_point = None
        self.have_changed = True
        
    def clearSelectPoint(self):
        self.selected_point = None
    
    def clearLabeledPoint(self):
        self.labeled_pts = []
        
    def drawSelectPoint(self, painter: QtGui.QPainter, pen_pt, m_ZoomValue, width, height, imgscale):
        if self.selected_point is None:
            return
        
        painter.setPen(pen_pt)
        r = 3 / m_ZoomValue
        cx = self.selected_point.x() * imgscale - width/2 - r
        cy = self.selected_point.y() * imgscale - height/2 - r
        draw_pt = QRectF(cx, cy, 2 * r, 2 * r)
        painter.drawEllipse(draw_pt)
    
    def drawLabeledPoints(self, painter: QtGui.QPainter, pen_pt, m_ZoomValue, width, height, imgscale):
        if len(self.labeled_pts) == 0:
            return
        
        painter.setPen(pen_pt)
        for each_pt in self.labeled_pts:
            r = 3 / m_ZoomValue
            cx = each_pt.x() * imgscale - width/2 - r
            cy = each_pt.y() * imgscale - height/2 - r
            draw_pt = QRectF(cx, cy, 2 * r, 2 * r)
            painter.drawEllipse(draw_pt)
            
    def drawLabelPointsOnImage(self, imgC, color_point, offset = None):
        if len(self.labeled_pts) == 0:
            return
        if offset is None:
            offset = np.array([0, 0])
        for each_Pt in self.labeled_pts:
            cx = int(each_Pt.x() + 0.5 + offset[0])
            cy = int(each_Pt.y() + 0.5 + offset[1])
            cv2.circle(imgC, (cx, cy), 2, color_point, 2)
        
            
    def add_selected_elp_pt(self, pt, fit_circle = False):
        self.selected_elp_pts.append(pt)
        
        if len(self.selected_elp_pts) > 5:
            pts = cvtQPtfs2pts(self.selected_elp_pts)
            gelp = ElliFit(pts, 'shape_image', fitCircle=fit_circle)
            self.selected_elp_shape = gelp
    
    def add_selected_elp(self):
        if self.selected_elp_shape is not None:
            self.labeled_elps.append({'gelp': self.selected_elp_shape,
                                      'pts': self.selected_elp_pts})
            self.clearSelectElp()
            self.have_changed = True
    
    def clearSelectElp(self):
        self.selected_elp_shape = None
        self.selected_elp_pts = []
        
    def clearLabeledElps(self):
        self.labeled_elps = []
        
    def drawSelectElp(self, painter: QtGui.QPainter, pen_pt, pen_elp, m_ZoomValue, width, height, imgscale):
        if len(self.selected_elp_pts) > 0:
            painter.setPen(pen_pt)
            for each_pt in self.selected_elp_pts:     
                r = 3 / m_ZoomValue
                cx = each_pt.x() * imgscale - width/2 - r
                cy = each_pt.y() * imgscale - height/2 - r
                draw_pt = QRectF(cx, cy, 2 * r, 2 * r)
                painter.drawEllipse(draw_pt)
        
        if self.selected_elp_shape is not None: # 椭圆拟合并绘制
            gelp = self.selected_elp_shape
                
            if gelp is not None:
                x, y = gelp.GenerateElpData(format_img=True)
                pts = np.vstack([x,y]).transpose()
                qptfs = cvtpts2Qptfs(pts)
                                
                for p1, p2 in zip(qptfs[:-1], qptfs[1:]):
                    painter.setPen(pen_elp)
                    dp1 = QPointF(p1.x() * imgscale - width/2, p1.y() * imgscale - height/2)
                    dp2 = QPointF(p2.x() * imgscale - width/2, p2.y() * imgscale - height/2)
                    linef = QLineF(dp1, dp2)
                    painter.drawLine(linef)
    
    def drawLabelElps(self, painter:QtGui.QPainter, pen_pt, pen_elp, m_ZoomValue, width, height, imgscale):
        if len(self.labeled_elps) > 0:
            for each_labeled_elp in self.labeled_elps:
                gelp = each_labeled_elp['gelp']
                pts = each_labeled_elp['pts']
                
                x, y = gelp.GenerateElpData(format_img=True)
                elp_pts = np.vstack([x,y]).transpose()
                qptfs = cvtpts2Qptfs(elp_pts)
                for p1, p2 in zip(qptfs[:-1], qptfs[1:]):
                    painter.setPen(pen_elp)
                    dp1 = QPointF(p1.x() * imgscale - width/2, p1.y() * imgscale - height/2)
                    dp2 = QPointF(p2.x() * imgscale- width/2, p2.y() * imgscale - height/2)
                    linef = QLineF(dp1, dp2)
                    painter.drawLine(linef)
                
                if len(pts) > 0:
                    painter.setPen(pen_pt)
                    for each_pt in pts:
                        r = 3 / m_ZoomValue
                        cx = each_pt.x() * imgscale - width/2 - r
                        cy = each_pt.y() * imgscale - height/2 - r
                        draw_pt = QRectF(cx, cy, 2 * r, 2 * r)
                        painter.drawEllipse(draw_pt)
                        
    def drawLabelElpsOnImage(self, imgC, color_ellipse, color_pts, thickness=2, offset = None):
        for each_labeled_elp in self.labeled_elps:
            gelp = each_labeled_elp['gelp']
            pts = each_labeled_elp['pts']
            if offset is not None:
                elp_data = gelp.ellipse_shape_img()
                elp_data[0:2] += offset
                gelp = GeneralEllipse(elp_data=elp_data, elp_type='shape_image')
            else:
                offset = np.array([0, 0])
            gelp.drawEllipse(imgC, color_ellipse, thickness)
            for each_pt in pts:
                cx = int(each_pt.x() + 0.5 + offset[0])
                cy = int(each_pt.y() + 0.5 + offset[1])
                cv2.circle(imgC, (cx, cy), 2, color_pts, thickness)
            
            
    
        
        
        
        
    