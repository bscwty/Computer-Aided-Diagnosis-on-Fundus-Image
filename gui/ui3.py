# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(798, 803)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(50, 200, 701, 541))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.LeftFrame = QtWidgets.QFrame(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.LeftFrame.sizePolicy().hasHeightForWidth())
        self.LeftFrame.setSizePolicy(sizePolicy)
        self.LeftFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.LeftFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.LeftFrame.setObjectName("LeftFrame")
        self.ResultList = QtWidgets.QListWidget(self.LeftFrame)
        self.ResultList.setGeometry(QtCore.QRect(50, 290, 221, 221))
        self.ResultList.setObjectName("ResultList")
        self.OrgImgLabel = QtWidgets.QLabel(self.LeftFrame)
        self.OrgImgLabel.setGeometry(QtCore.QRect(0, 0, 121, 31))
        self.OrgImgLabel.setObjectName("OrgImgLabel")
        self.DiseaseLabel = QtWidgets.QLabel(self.LeftFrame)
        self.DiseaseLabel.setGeometry(QtCore.QRect(0, 260, 121, 31))
        self.DiseaseLabel.setObjectName("DiseaseLabel")
        self.OrgImg = QtWidgets.QLabel(self.LeftFrame)
        self.OrgImg.setGeometry(QtCore.QRect(60, 30, 224, 224))
        self.OrgImg.setText("")
        self.OrgImg.setObjectName("OrgImg")
        self.horizontalLayout.addWidget(self.LeftFrame)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout.addItem(spacerItem)
        self.frame = QtWidgets.QFrame(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.HeatmapLabel = QtWidgets.QLabel(self.frame)
        self.HeatmapLabel.setGeometry(QtCore.QRect(0, 0, 121, 31))
        self.HeatmapLabel.setObjectName("HeatmapLabel")
        self.DiseaseIntro = QtWidgets.QLabel(self.frame)
        self.DiseaseIntro.setGeometry(QtCore.QRect(50, 300, 221, 131))
        self.DiseaseIntro.setText("")
        self.DiseaseIntro.setObjectName("DiseaseIntro")
        self.HeatmapImg = QtWidgets.QGraphicsView(self.frame)
        self.HeatmapImg.setGeometry(QtCore.QRect(50, 30, 224, 224))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.HeatmapImg.sizePolicy().hasHeightForWidth())
        self.HeatmapImg.setSizePolicy(sizePolicy)
        self.HeatmapImg.setObjectName("HeatmapImg")
        self.DiseaseIntroLabel = QtWidgets.QLabel(self.frame)
        self.DiseaseIntroLabel.setGeometry(QtCore.QRect(0, 260, 121, 31))
        self.DiseaseIntroLabel.setObjectName("DiseaseIntroLabel")
        self.HeatmapImg_2 = QtWidgets.QGraphicsView(self.frame)
        self.HeatmapImg_2.setGeometry(QtCore.QRect(50, 290, 224, 224))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.HeatmapImg_2.sizePolicy().hasHeightForWidth())
        self.HeatmapImg_2.setSizePolicy(sizePolicy)
        self.HeatmapImg_2.setObjectName("HeatmapImg_2")
        self.horizontalLayout.addWidget(self.frame)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(50, 130, 701, 61))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.MiddleLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.MiddleLayout.setContentsMargins(0, 0, 0, 0)
        self.MiddleLayout.setObjectName("MiddleLayout")
        self.CAM = QtWidgets.QRadioButton(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.CAM.sizePolicy().hasHeightForWidth())
        self.CAM.setSizePolicy(sizePolicy)
        self.CAM.setObjectName("CAM")
        self.MiddleLayout.addWidget(self.CAM)
        self.GradCAM = QtWidgets.QRadioButton(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.GradCAM.sizePolicy().hasHeightForWidth())
        self.GradCAM.setSizePolicy(sizePolicy)
        self.GradCAM.setObjectName("GradCAM")
        self.MiddleLayout.addWidget(self.GradCAM)
        self.MiddleFrame = QtWidgets.QFrame(self.horizontalLayoutWidget_2)
        self.MiddleFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.MiddleFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.MiddleFrame.setObjectName("MiddleFrame")
        self.MiddleLayout.addWidget(self.MiddleFrame)
        self.GradCAMpp = QtWidgets.QRadioButton(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.GradCAMpp.sizePolicy().hasHeightForWidth())
        self.GradCAMpp.setSizePolicy(sizePolicy)
        self.GradCAMpp.setObjectName("GradCAMpp")
        self.MiddleLayout.addWidget(self.GradCAMpp)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.MiddleLayout.addItem(spacerItem1)
        self.OpenFile = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.OpenFile.sizePolicy().hasHeightForWidth())
        self.OpenFile.setSizePolicy(sizePolicy)
        self.OpenFile.setObjectName("OpenFile")
        self.MiddleLayout.addWidget(self.OpenFile)
        self.GradCAM.raise_()
        self.GradCAMpp.raise_()
        self.OpenFile.raise_()
        self.CAM.raise_()
        self.MiddleFrame.raise_()
        self.AboveFrame = QtWidgets.QFrame(self.centralwidget)
        self.AboveFrame.setGeometry(QtCore.QRect(50, 30, 701, 101))
        self.AboveFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.AboveFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.AboveFrame.setObjectName("AboveFrame")
        self.Icon = QtWidgets.QLabel(self.AboveFrame)
        self.Icon.setGeometry(QtCore.QRect(120, 10, 91, 91))
        self.Icon.setText("")
        self.Icon.setObjectName("Icon")
        self.Title = QtWidgets.QLabel(self.AboveFrame)
        self.Title.setGeometry(QtCore.QRect(220, 20, 361, 51))
        font = QtGui.QFont()
        font.setFamily("华文细黑")
        font.setPointSize(18)
        self.Title.setFont(font)
        self.Title.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.Title.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.Title.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Title.setTextFormat(QtCore.Qt.PlainText)
        self.Title.setObjectName("Title")
        self.BackLabel = QtWidgets.QLabel(self.centralwidget)
        self.BackLabel.setGeometry(QtCore.QRect(50, 200, 701, 541))
        self.BackLabel.setText("")
        self.BackLabel.setObjectName("BackLabel")
        self.AboveFrame.raise_()
        self.horizontalLayoutWidget.raise_()
        self.horizontalLayoutWidget_2.raise_()
        self.BackLabel.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 798, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setObjectName("action")
        self.action_2 = QtWidgets.QAction(MainWindow)
        self.action_2.setObjectName("action_2")
        self.menu.addAction(self.action)
        self.menu.addAction(self.action_2)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "眼底多病种识别工具"))
        self.OrgImgLabel.setText(_translate("MainWindow", "原始图片："))
        self.DiseaseLabel.setText(_translate("MainWindow", "预测结果："))
        self.HeatmapLabel.setText(_translate("MainWindow", "分割结果："))
        self.DiseaseIntroLabel.setText(_translate("MainWindow", "预测结果："))
        self.CAM.setText(_translate("MainWindow", "CAM"))
        self.GradCAM.setText(_translate("MainWindow", "GradCAM"))
        self.GradCAMpp.setText(_translate("MainWindow", "GradCAM++"))
        self.OpenFile.setText(_translate("MainWindow", "打开图片"))
        self.Title.setText(_translate("MainWindow", "东南大学 眼底多病种识别"))
        self.menu.setTitle(_translate("MainWindow", "数据集及模型信息"))
        self.action.setText(_translate("MainWindow", "多病种分类"))
        self.action_2.setText(_translate("MainWindow", "血管视盘分割"))
