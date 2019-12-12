import sys
from PySide2.QtWidgets import *
from PySide2 import QtGui
from PySide2.QtCore import Qt
from widgets.standard_select import StandardSelect

from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import pandas as pd


class Calibration(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.mainWidget = QWidget()
        self.mainLayout = QVBoxLayout(self.mainWidget)
        self.mainLayout.setAlignment(Qt.AlignCenter)

        self.settings = QWidget()
        self.settingsLayout = QHBoxLayout(self.settings)
        self.stdSelect = StandardSelect(self)
        self.stdSelect.setFixedHeight(200)
        self.settingsLayout.addWidget(self.stdSelect)

        self.formGroupBox = QGroupBox("")
        layout = QFormLayout()
        self.equationBtn = QPushButton('Calculate')
        layout.addRow(QLabel('Cal. Equations:'), self.equationBtn)
        self.equationBtn.clicked.connect(self.equations)
        self.exportBtn = QPushButton('Export')
        layout.addRow(QLabel('Export Equations:'), self.exportBtn)
        self.exportBtn.clicked.connect(self.export_equations)
        self.elem1 = QComboBox()
        self.elem1.currentTextChanged.connect(self.graph)
        layout.addRow(QLabel("Graph 1:"), self.elem1)
        self.elem2 = QComboBox()
        self.elem2.currentTextChanged.connect(self.graph)
        layout.addRow(QLabel("Graph 2:"), self.elem2)
        self.formGroupBox.setLayout(layout)
        self.plotBtn = QPushButton('Plot')
        layout.addRow(QLabel('Show graphs:'), self.plotBtn)
        self.plotBtn.clicked.connect(self.graph)

        self.settingsLayout.addWidget(self.formGroupBox)
        self.settingsLayout.addStretch(1)
        self.settingsLayout.setSizeConstraint(QLayout.SetFixedSize)
        self.mainLayout.addWidget(self.settings)

        self.fig = Figure()
        self.ax = self.fig.subplots(1,2)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("QWidget {border: None; background-color: white; color: black}")
        self.mainLayout.addWidget(self.toolbar)
        self.mainLayout.addWidget(self.canvas)

    def get_page(self):
        """
        Returns Page for bulk analysis for main stack of gui
        Returns
        -------
        mainWidget : QWidget
            The widget that the houses the bulk analysis screen.
        """
        return self.mainWidget

    def equations(self):
        stds = self.stdSelect.return_all()
        self.parent.Data.get_regression_values('intensity', stds)
        self.parent.Data.calibration_equations()

    def graph(self):
        if not self.parent.Data.regression_equations:
            return

        elem1 = self.elem1.currentText()
        elem2 = self.elem2.currentText()
        self.parent.Data.calibration_graph(elem1, ax=self.ax[0])
        self.parent.Data.calibration_graph(elem2, ax=self.ax[1])
        self.canvas.draw_idle()

    def export_equations(self):

        if self.parent.Data.regression_equations:
            regr_eq = pd.DataFrame.from_dict(self.parent.Data.regression_equations)
            #regr_eq.index = ['intercept', 'slope']
            filename, filters = QFileDialog.getSaveFileName(self, caption='Save file', dir='.')
        else:
            msg_dialog = QMessageBox()
            msg_dialog.setIcon(QMessageBox.Information)
            msg_dialog.setText('No data to save.')
            msg_dialog.exec_()
            return

        if filename:
            writer = pd.ExcelWriter(filename, engine='xlsxwriter')
            regr_eq.to_excel(writer, sheet_name='regression')
            writer.save()

