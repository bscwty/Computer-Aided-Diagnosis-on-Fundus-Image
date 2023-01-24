import sys
from PyQt5.QtWidgets import QApplication, QMainWindow


from gui.mainwindow import  Ui_MainWindow


from seg.api import Seg
from cla.api import Cla


if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    mainwindow = QMainWindow()

    vess_seg_model_path = './saved_models/vess_model.pth'
    disc_seg_model_path = './saved_models/disc_model.pth'
    cla_model_path = './saved_models/clas_model.pkl'

    seg = Seg(vess_seg_model_path, disc_seg_model_path)
    cla = Cla(cla_model_path)

    ui = Ui_MainWindow(mainwindow)
    mainwindow.show()

    ui.load(seg, cla)

    sys.exit(app.exec_())

'''
TODO:
5. 多标记 -> 多病种 / 英文疾病 翻译 成中文 ?
6. 创建展示用数据集（50张 9类 最好多疾病）
'''
    
