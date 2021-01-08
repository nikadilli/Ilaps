import sys
from PySide2.QtWidgets import *
from PySide2 import QtGui
from PySide2.QtCore import Qt

from DataClass import MSData
from widgets.pandas_model import PandasModel

from xlrd import XLRDError


class Welcome(QWidget):
    """
    Starting screen of Ilaps GUI, allowing importing analysis data

    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.mainWidget = QWidget()
        self.mainLayout = QVBoxLayout(self.mainWidget)
        self.mainLayout.setAlignment(Qt.AlignCenter)

        self.importAnalysis = QGroupBox('Import Data')
        self.mainLayout.addWidget(self.importAnalysis)
        self.analysisLayout = QGridLayout()

        self.lbl = QLabel('Import Analysis')
        self.analysisLayout.addWidget(self.lbl, 0, 0)

        self.loadData1 = QPushButton('')
        self.loadData1.setIcon(QtGui.QIcon('./imgs/load.jpg'))
        self.analysisLayout.addWidget(
            self.loadData1, 1, 0, alignment=Qt.AlignLeft)
        self.loadData1.clicked.connect(lambda: self.get_file(self.dataEntry1))

        self.dataEntry1 = QLineEdit()
        self.analysisLayout.addWidget(
            self.dataEntry1, 1, 1, alignment=Qt.AlignLeft)

        self.lbl = QLabel('File type:')
        self.analysisLayout.addWidget(self.lbl, 2, 0)

        self.fileType = QComboBox()
        self.fileType.addItems(['csv', 'xlsx', 'asc'])
        self.analysisLayout.addWidget(self.fileType, 2, 1)

        self.lbl = QLabel('Instrument type:')
        self.analysisLayout.addWidget(self.lbl, 3, 0)

        self.instrument = QComboBox()
        self.instrument.addItems(['raw', 'Agilent', 'Element'])
        self.analysisLayout.addWidget(self.instrument, 3, 1)

        self.lbl = QLabel('Import Parameters')
        self.analysisLayout.addWidget(self.lbl, 4, 0)

        self.loadData2 = QPushButton('')
        self.loadData2.setIcon(QtGui.QIcon('./imgs/load.jpg'))
        self.analysisLayout.addWidget(
            self.loadData2, 5, 0, alignment=Qt.AlignLeft)
        self.loadData2.clicked.connect(lambda: self.get_file(self.dataEntry2))

        self.dataEntry2 = QLineEdit()
        self.analysisLayout.addWidget(
            self.dataEntry2, 5, 1, alignment=Qt.AlignLeft)

        self.lbl = QLabel('Import Iolite')
        self.analysisLayout.addWidget(self.lbl, 6, 0)

        self.loadIolite = QPushButton('')
        self.loadIolite.setIcon(QtGui.QIcon('./imgs/load.jpg'))
        self.analysisLayout.addWidget(
            self.loadIolite, 7, 0, alignment=Qt.AlignLeft)
        self.loadIolite.clicked.connect(lambda: self.get_file(self.dataEntry3))

        self.dataEntry3 = QLineEdit()
        self.analysisLayout.addWidget(
            self.dataEntry3, 7, 1, alignment=Qt.AlignLeft)

        self.importAnalysis.setLayout(self.analysisLayout)

        self.importBtn = QPushButton('Import')
        self.mainLayout.addWidget(self.importBtn)
        self.importBtn.clicked.connect(self.import_data)

    def get_page(self):
        """
        Returns Page for bulk analysis for main stack of gui
        Returns
        -------
        mainWidget : QWidget
            The widget that the houses the bulk analysis screen.
        """
        return self.mainWidget

    def get_file(self, entry):
        """
        Fill lineEdit with path to file from user
        """
        filename, filters = QFileDialog.getOpenFileName(self, caption='Open file', dir='',
                                                        filter='All Files(*.*);; Excel Files(*.xlsx)')
        if filename:
            entry.setText(filename)

    def import_data(self):
        """
        Create MSData object from files holding all analysis data
        """
        if self.parent.Data is not None:
            self.parent.Data = None

        try:
            self.parent.Data = MSData(self.dataEntry1.text(),
                                      filetype=str(
                                          self.fileType.currentText()),
                                      instrument=str(self.instrument.currentText()))
        except Exception as e:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('Data not imported. <br>{}'.format(e))
            error_dialog.exec_()
            return

        if self.dataEntry2.text():
            try:
                self.parent.Data.read_param(str(self.dataEntry2.text()))
            except FileNotFoundError as e:
                error_dialog = QErrorMessage()
                error_dialog.showMessage('{}'.format(e))
                error_dialog.exec_()
                return
            except XLRDError as e:
                error_dialog = QErrorMessage()
                error_dialog.showMessage('{}'.format(e))
                error_dialog.exec_()
                return

        if self.dataEntry3.text():
            try:
                self.parent.Data.read_iolite(str(self.dataEntry3.text()))
                print(self.parent.Data.iolite)
            except FileNotFoundError as e:
                error_dialog = QErrorMessage()
                error_dialog.showMessage('{}'.format(e))
                error_dialog.exec_()
                return
            except XLRDError as e:
                error_dialog = QErrorMessage()
                error_dialog.showMessage('{}'.format(e))
                error_dialog.exec_()
                return

        stds = self.parent.Data.srms.index.values
        self.parent.bulk_analysis.standard.clear()
        self.parent.bulk_analysis.standard.addItems(stds)
        self.parent.calibration.stdSelect.stdList.clear()
        self.parent.calibration.stdSelect.stdList.addItems(stds)

        elems = self.parent.Data.elements
        self.parent.imaging.element.clear()
        self.parent.imaging.element.addItems(elems)
        self.parent.calibration.elem1.clear()
        self.parent.calibration.elem1.addItems(elems)
        self.parent.calibration.elem2.clear()
        self.parent.calibration.elem2.addItems(elems)
        self.parent.show_data.elem.clear()
        self.parent.show_data.elem.addItems(elems)
        self.parent.Data.graph(ax=self.parent.data_select.ax)
        self.parent.data_select.canvas.draw()

        # for tab in self.parent.bulk_analysis.table.tabLst.values():
        # self.parent.bulk_analysis.table.table[tab].clearSpans()

        model = PandasModel(self.parent.Data.data)
        self.parent.bulk_analysis.table.table['Raw'].setModel(model)

        self.parent.change_layout(1)
