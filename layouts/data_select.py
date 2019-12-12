import sys
from PySide2.QtWidgets import *
from PySide2 import QtGui
from PySide2.QtCore import Qt
from widgets.tab_select import SelectTab

import numpy as np

from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from widgets.peak_select import PeakSelect
from side_functions import *


class DataSelect(QWidget):

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.mainWidget = QWidget()
        self.mainLayout = QVBoxLayout(self.mainWidget)
        self.mainLayout.setAlignment(Qt.AlignCenter)

        self.settingsWidget = QWidget()
        self.settingsLayout = QHBoxLayout(self.settingsWidget)

        self.mainLayout.addWidget(self.settingsWidget)

        self.formGroupBox = QGroupBox("Analysis settings")
        layout = QFormLayout()

        self.ablation = QLineEdit()
        layout.addRow(QLabel("Peak ablation time:"), self.ablation)
        self.ablation.returnPressed.connect(self.set_abl_time)

        self.skipBcg = QHBoxLayout()
        self.skipBcgStart = QLineEdit()
        self.skipBcgEnd = QLineEdit()
        self.skipBcg.addWidget(self.skipBcgStart)
        self.skipBcg.addWidget(self.skipBcgEnd)
        layout.addRow(QLabel("Skip background:"), self.skipBcg)

        self.skipPeak = QHBoxLayout()
        self.skipPeakStart = QLineEdit()
        self.skipPeakEnd = QLineEdit()
        self.skipPeak.addWidget(self.skipPeakStart)
        self.skipPeak.addWidget(self.skipPeakEnd)
        layout.addRow(QLabel("Skip peak:"), self.skipPeak)

        self.method = QComboBox()
        self.method.addItems(['area', 'intensity'])
        layout.addRow(QLabel("Method:"), self.method)

        self.filter = QComboBox()
        self.filter.addItems(['sum'])
        layout.addRow(QLabel("Filter line:"), self.filter)

        self.formGroupBox.setLayout(layout)

        self.settingsLayout.addWidget(self.formGroupBox)

        self.select = SelectTab(self)
        self.settingsLayout.addWidget(self.select)

        self.selectBtn = QPushButton('Select')
        self.settingsLayout.addWidget(self.selectBtn, alignment=Qt.AlignCenter)
        self.selectBtn.clicked.connect(self.select_data)

        self.peakSelect = PeakSelect(self)
        self.settingsLayout.addWidget(self.peakSelect)

        self.settingsLayout.setSizeConstraint(QLayout.SetFixedSize)
        #self.settingsLayout.addStretch(1)

        self.fig = Figure()
        self.ax = self.fig.subplots()
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

    def set_abl_time(self):
        """
        Set time of ablation for MSData
        """
        time = int(self.ablation.text())
        self.parent.Data.set_ablation_time(time)

    def select_data(self):
        filt = self.filter.currentText()
        self.parent.Data.set_filtering_element(filt)

        if is_digit(self.skipBcgStart.text()):
            bcg_s = int(self.skipBcgStart.text())
        else:
            bcg_s = 0

        if is_digit(self.skipBcgEnd.text()):
            bcg_e = int(self.skipBcgEnd.text())
        else:
            bcg_e = 0

        if is_digit(self.skipPeakStart.text()):
            sig_s = int(self.skipPeakStart.text())
        else:
            sig_s = 0

        if is_digit(self.skipPeakEnd.text()):
            sig_e = int(self.skipPeakEnd.text())
        else:
            sig_e = 0

        self.parent.Data.set_skip(bcg_s, bcg_e, sig_s, sig_e)

        method = self.select.return_tab()

        if method == 'Treshold':
            time = float(self.select.bcg.text())
            multiply = float(self.select.mltp.text())
            try:
                self.parent.Data.create_selector_bcg(multiply, time)
            except IndexError as e:
                error_dialog = QErrorMessage()
                error_dialog.showMessage('Data not selected. <br>{}'.format(e))
                error_dialog.exec_()

        if method == 'Iolite':
            if self.parent.Data.iolite.empty:
                error_dialog = QErrorMessage()
                error_dialog.showMessage('Missing Iolite file.')
                error_dialog.exec_()
                return

            start = float(self.select.start.text())
            self.parent.Data.create_selector_iolite(start)

        if method =='Gradient':
            time = float(self.select.time.text())
            self.parent.Data.create_selector_gradient(time)

        self.ax.clear()
        self.ax.figure.canvas.draw_idle()

        try:
            self.parent.Data.graph(ax=self.ax)
            pass
        except IndexError as e:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('Data not selected. <br>{}'.format(e))
            error_dialog.exec_()

        self.ax.figure.canvas.draw_idle()

        self.peakSelect.peakList.clear()
        if self.parent.Data.names:
            self.peakSelect.peakList.addItems(self.parent.Data.names)
        else:
            self.parent.Data.names = ['peak_{}'.format(i) for i in range(1, len(self.parent.Data.laser_on) + 1)]
            self.peakSelect.peakList.addItems(self.parent.Data.names)

