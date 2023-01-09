# 标记工具PoseLabel的使用说明





## 补充一个自定义形状


为了方便扩展其他形状，程序具有一定程度的模块能力，按照如下操作即可添加自定义形状

### (1) 在**Shape.py**中添加一个形状信息

每个步骤按照添加椭圆形状示例说明。

1. 补充初始化部分的变量。包含4个部分：当前操作点(`self.selected_elp_pts`)，由当前操作点确定的操作形状(`self.selected_elp_shape`)，已确认标记的形状(`self.labeled_elps`)，形状名(`self.name_ellipse`)。
2. 补充动态选点的函数，表示当前操作的点，并动态更新形状参数
3. 补充确认当前操作形状的函数，
4. 补充清空当前操作形状和清空所有已选形状的函数
5. 补充绘制当前操作形状和绘制已标记形状的函数
6. 在saveJson和loadJson中补充当前形状io的部分，如果存在自定义类，在ShapeEncoder中补充对应的功能即可
7. 在clearAll中补充清空所有数据的功能



### (2) 在**ImageView.py**中添加案件响应信息

补充一个新的形状类型需要补充的部分
1. 在初始化中补充一个启动标记变量
2. 模拟`onLabelEllipse`补充标记启动关闭功能
3. 在'contextMenuEvent'中补充菜单Action 
4. 在`mouseDoubleClickEvent`补充双击选点功能
5. 在`keyPressEvent`中补充确认和取消功能
6. 在`paintEvent`中补充当前选择形状和已选择形状的绘制功能


### (2) 在**ImageView.py**中添加界面操作的功能

