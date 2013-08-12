import os
import sys
import zlib

import numpy as np
import matplotlib

# matplotlib.rcParams['backend.qt4'] = 'PySide'
from PySide import QtCore, QtGui

# from matplotlib.backends.backend_qt4 import FigureCanvasQT as FigureCanvas
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import sl.objects.conventions as conv

from sl.lib import plotlib, tinylib

_image_dir = os.path.join(os.path.dirname(__file__), 'images/')

class WindWidget(FigureCanvas):

    def load(self, fcst):
        self.index = 0
        self.bmap = plotlib.draw_map(fcst, self.axes)
        self.background = self.figure.canvas.copy_from_bbox(self.figure.bbox)
        self.wind_map = plotlib.WindMap(fcst.take([self.index], conv.TIME), self.bmap)

    def __init__(self, fcst, parent=None):
        super(WindWidget, self).__init__(Figure())
        self.setParent(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.fcst = fcst
        self.load(fcst)

    def Plot(self, index=None):
        self.index = index or self.index
        self.figure.canvas
        self.wind_map.update(self.fcst.take([self.index], conv.TIME))
        self.figure.canvas.blit(self.axes.bbox)
        self.figure.canvas.draw()
        import pdb; pdb.set_trace()

    def next_forecast(self):
        if self.index < (self.fcst.dimensions[conv.TIME] - 1):
            self.Plot(self.index + 1)

class HoverButton(QtGui.QPushButton):
    _arrow_left = _image_dir + 'arrow_left_50px_transparent.png'
    _arrow_left_hover = _image_dir + 'arrow_left_50px.png'
    _arrow_right = _image_dir + 'arrow_right_50px_transparent.png'
    _arrow_right_hover = _image_dir + 'arrow_right_50px.png'

    def __init__(self, parent=None, image=None):
        super(HoverButton, self).__init__(parent)

        if image:
            style = """
                QPushButton{
                  background:url(%s); 
                  background-repeat:no-repeat;
                  background-position:center;
                  background-color:rgba( 0, 0, 0, 0%% ); 
                  border: 0px}
                QPushButton:hover{
                  background-color:rgba( 0, 0, 0, 75%% ); 
                  border: 0px}"""
            style = style % image
        else:
            style = ("QPushButton{background-color:rgba( 0, 0, 0, 0% ); border: 0px}" +
                     "QPushButton:hover{background-color:rgba( 0, 0, 0, 50%% ); border: 0px}")

        self.setFlat(True)
        self.setAutoFillBackground(True)
        self.setStyleSheet(style)
        self.setMouseTracking(True)

class MatplotlibWidget(FigureCanvas):

    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(Figure())
        self.setParent(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.data = np.random.random((10, 10))
        self.data = np.zeros((10, 10))
        self.data[2:3, :] = 1
        self.axes.imshow(self.data)

    def Plot(self):
        self.data = np.random.random((10, 10))
        self.axes.imshow(self.data)

# the majority of the following class comes from using 'pyside-uic blah.ui'
# where blah.ui is a simple gui made in the qt designer

class Ui_MainWindow(object):

    def setupUi(self, MainWindow, fcst=None):
        MainWindow.setObjectName("Slocum - Forecast Viewer")
        MainWindow.resize(600, 625)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.widget = WindWidget(fcst, self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(0, 25, 600, 600))
        self.widget.setObjectName("widget")
        self.widget.mpl_connect('button_press_event', self.status_update)

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtGui.QMenuBar()
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 25))
        self.menubar.setObjectName("menubar")

        fileopen = QtGui.QAction("Open", MainWindow)
        fileopen.setShortcut('Ctrl+O')
        fileopen.setStatusTip("Open a New Forecast")
        fileopen.triggered.connect(self.action_fileopen)

        self.filemenu = self.menubar.addMenu("&File")
        self.filemenu.addAction(fileopen)

        self.menubar.addAction(self.filemenu.menuAction())

        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.prev_button = HoverButton(MainWindow, _image_dir + 'arrow_left_50px.png')
        self.prev_button.setGeometry(QtCore.QRect(0, 25, 50, 575))
        self.prev_button.setObjectName("pushButton")
        self.prev_button.connect(QtCore.SIGNAL("clicked()"), self.next_forecast)

        self.next_button = HoverButton(MainWindow, _image_dir + 'arrow_right_50px.png')
        self.next_button.setGeometry(QtCore.QRect(550, 25, 50, 575))
        self.next_button.setObjectName("pushButton")
        self.next_button.connect(QtCore.SIGNAL("clicked()"), self.next_forecast)

#        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def next_forecast(self):
        self.statusbar.showMessage("Loading next forecast")
        self.widget.next_forecast()
        self.statusbar.showMessage("")

    def action_fileopen(self):
        ret = QtGui.QFileDialog.getOpenFileName(self.widget, "Open Forecast", "", "Windbreakers (*.fcst)")
        if len(ret[0]):
            f = open(ret[0], 'r')
            unzipped = zlib.decompress(f.read())
            fcst = tinylib.from_beaufort(unzipped)
            self.widget.load(fcst)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton.setText(QtGui.QApplication.translate("MainWindow", "clickme", None, QtGui.QApplication.UnicodeUTF8))

    # the rest of this class was hand coded
        self.pushButton.clicked.connect(self.output)
        self.count = 0

    def status_update(self, event):
        self.statusbar.showMessage(str(int(0.5 + event.ydata)))


def main():
    f = open('/home/kleeman/dev/slocum/data/windbreaker.fcst', 'r')
    unzipped = zlib.decompress(f.read())
    fcst = tinylib.from_beaufort(unzipped)

    try:
        app = QtGui.QApplication(sys.argv)
    except RuntimeError:
        pass

    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow, fcst)
    MainWindow.show()
    app.exec_()

if __name__ == "__main__":
    sys.exit(main())
