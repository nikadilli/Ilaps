
from PySide2.QtWidgets import *
from PySide2.QtCore import Qt
from style import IlapsStyle


class EndLoop(Exception):
    pass


class ElemWindow(QMainWindow):
    def __init__(self, parent):
        super(ElemWindow, self).__init__(parent)

        self.parent = parent

        # self.setGeometry(250, 250, 250, 250)
        self.setWindowTitle("Elements selection")
        self.setStyleSheet(IlapsStyle)

        self.mainWidget = QWidget()
        self.setCentralWidget(self.mainWidget)
        self.mainLayout = QVBoxLayout(self.mainWidget)

        self.elemWidget = QWidget()
        self.elemLayout = QGridLayout(self.elemWidget)
        self.mainLayout.addWidget(self.elemWidget)

        if self.parent.parent.Data is not None:
            elems_nr = len(self.parent.parent.Data.elements)

            self.elemsChBxs = []

            try:
                for i in range(10):
                    for j in range(10):
                        idx = i*10 + j
                        if idx == elems_nr - 1:
                            raise EndLoop
                        self.elemsChBxs.append(QCheckBox(self.parent.parent.Data.elements[idx]))
                        self.elemLayout.addWidget(self.elemsChBxs[idx], i, j)
            except EndLoop:
                pass
        else:
            self.elemLayout.addWidget(QLabel('Analysis not imported.'))

        self.buttonWidget = QWidget()
        self.buttonLayout = QHBoxLayout(self.buttonWidget)
        self.mainLayout.addWidget(self.buttonWidget)

        self.buttonLayout.addStretch(1)
        self.cancelBtn = QPushButton('Cancel')
        self.buttonLayout.addWidget(self.cancelBtn, alignment=Qt.AlignRight)
        self.cancelBtn.clicked.connect(self.cancel)

        self.selectBtn = QPushButton('Remove')
        self.buttonLayout.addWidget(self.selectBtn, alignment=Qt.AlignRight)
        self.selectBtn.clicked.connect(self.action)

    def action(self):
        cols_to_drop = [el.text() for el in self.elemsChBxs if el.isChecked()]
        self.parent.parent.Data.data = self.parent.parent.Data.data.drop(columns=cols_to_drop)
        self.parent.parent.Data.elements = list(self.parent.parent.Data.data.columns)
        self.close()

    def cancel(self):
        self.close()
