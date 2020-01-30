import sys
from PySide2.QtWidgets import *
from PySide2 import QtGui, QtCore
from PySide2.QtCore import Qt

from DataClass import MSData

from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


class Imaging(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.mainWidget = QWidget()
        self.mainLayout = QHBoxLayout(self.mainWidget)
        self.mainLayout.setAlignment(Qt.AlignTop)

        self.settingsGroup = QGroupBox('')
        layout = QFormLayout()

        self.dx = QLineEdit()
        layout.addRow(QLabel("Delta x: "), self.dx)
        self.dy = QLineEdit()
        layout.addRow(QLabel("Delta y: "), self.dy)
        self.background = QComboBox()
        self.background.addItems(['begining', 'all'])
        layout.addRow('Background: ', self.background)
        layout.addRow(QLabel(""),)
        self.minz = QLineEdit()
        layout.addRow(QLabel("Min z: "), self.minz)
        self.maxz= QLineEdit()
        layout.addRow(QLabel("Max z: "), self.maxz)
        layout.addRow(QLabel(""), )
        self.intercept = QLineEdit()
        layout.addRow(QLabel("Intercept: "), self.intercept)
        self.slope = QLineEdit()
        layout.addRow(QLabel("Slope: "), self.slope)
        layout.addRow(QLabel(""), )
        self.units = QLineEdit()
        layout.addRow(QLabel("Units: "), self.units)
        self.title = QLineEdit()
        layout.addRow(QLabel("Title: "), self.title)
        layout.addRow(QLabel(""), )
        self.interpolation = QComboBox()
        self.interpolation.addItems(['nearest', 'bilinear', 'bicubic', 'spline16',
                                     'spline36', 'quadric', 'gaussian', 'lanczos'])
        self.interpolation.currentTextChanged.connect(self.image)
        layout.addRow('Interpolation: ', self.interpolation)
        self.cmap = QComboBox()
        self.cmap.addItems(['jet', 'grey', 'binary', 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
                            ])
        self.cmap.currentTextChanged.connect(self.image)
        layout.addRow('Colour map: ', self.cmap)
        layout.setSizeConstraint(QLayout.SetFixedSize)
        self.settingsGroup.setLayout(layout)
        self.mainLayout.addWidget(self.settingsGroup)

        self.settingsGroup2 = QGroupBox('')
        layout2 = QVBoxLayout()
        layout2.addStretch(5)

        self.matrixBtn = QPushButton('Create matrix')
        layout2.addWidget(self.matrixBtn)
        self.matrixBtn.clicked.connect(self.matrix)
        layout2.addStretch(1)

        self.element = QComboBox()
        layout2.addWidget(self.element)

        self.imageBtn = QPushButton('Image')
        layout2.addWidget(self.imageBtn)
        self.imageBtn.clicked.connect(self.image)

        self.rotateBtn = QPushButton('Rotate')
        layout2.addWidget(self.rotateBtn)
        self.rotateBtn.clicked.connect(self.rotate)

        self.quantifyBtn = QPushButton('Quantify')
        layout2.addWidget(self.quantifyBtn)
        self.quantifyBtn.clicked.connect(self.quantify)
        layout2.addStretch(1)
        self.importMatrixBtn = QPushButton('Import Matrix')
        layout2.addWidget(self.importMatrixBtn)
        self.importMatrixBtn.clicked.connect(self.import_matrix)
        layout2.addStretch(1)
        self.exportMatrixBtn = QPushButton('Export Matrix')
        layout2.addWidget(self.exportMatrixBtn)
        self.exportMatrixBtn.clicked.connect(self.export_matrix)
        self.quantifyAllBtn = QPushButton('Quantify all')
        self.quantifyAllBtn.clicked.connect(self.quantify_all)
        layout2.addWidget(self.quantifyAllBtn)
        self.exportQuantifiedBtn = QPushButton('Export Quantified')
        self.exportQuantifiedBtn.clicked.connect(self.export_quantified)
        layout2.addWidget(self.exportQuantifiedBtn)

        layout2.addStretch(5)
        self.settingsGroup2.setLayout(layout2)
        self.mainLayout.addWidget(self.settingsGroup2)

        self.graph = QWidget()
        self.graphLayout = QVBoxLayout(self.graph)
        self.fig = Figure()
        self.ax = self.fig.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("QWidget {border: None; background-color: white; color: black}")
        self.graphLayout.addWidget(self.toolbar)
        self.graphLayout.addWidget(self.canvas)

        self.mainLayout.addWidget(self.graph)

    def get_page(self):
        """
        Returns Page for bulk analysis for main stack of gui
        Returns
        -------
        mainWidget : QWidget
            The widget that the houses the bulk analysis screen.
        """
        return self.mainWidget

    def matrix(self):
        dx = int(self.dx.text())
        dy = int(self.dy.text())
        self.parent.Data.setxy(dx, dy)

        self.parent.Data.create_all_maps(self.background.currentText())

    def image(self, quantified=False):
        if not self.parent.Data.maps:
            msg_dialog = QMessageBox()
            msg_dialog.setIcon(QMessageBox.Information)
            msg_dialog.setText('No matrix to plot.')
            msg_dialog.exec_()
            return

        elem = self.element.currentText()
        interpolation = self.interpolation.currentText()
        cmap = self.cmap.currentText()

        if self.minz.text().isdigit():
            vmin = int(self.minz.text())
        else:
            vmin = None

        if self.maxz.text().isdigit():
            vmax = int(self.maxz.text())
        else:
            vmax = None

        title = self.title.text()
        units = self.units.text()

        self.fig.clear()
        self.ax = self.fig.subplots()
        self.ax.figure.canvas.draw_idle()


        self.parent.Data.elemental_image(fig=self.fig, ax=self.ax, elem=elem, vmin=vmin, vmax=vmax,
                                         interpolate=interpolation, colourmap=cmap, title=title,
                                         units=units, quantified=quantified)

        self.ax.figure.canvas.draw_idle()

    def rotate(self):
        elem = self.element.currentText()
        self.parent.Data.rotate_map(elem)

        self.image()

    def import_matrix(self):

        filename, filters = QFileDialog.getOpenFileName(self, caption='Open file', dir='.',
                                                        filter="Excel files (*.xlsx)")
        if not filename:
            return

        if self.parent.Data is not None:
            self.parent.Data = MSData()

        self.parent.Data.import_matrices(filename)
        elems = self.parent.Data.elements
        self.element.addItems(elems)
        self.parent.show_data.elem.addItems(elems)

        # msg_dialog = QMessageBox()
        # msg_dialog.setIcon(QMessageBox.Information)
        # msg_dialog.setText('Not supported yet.')
        # msg_dialog.exec_()

    def export_matrix(self):
        if self.parent.Data.maps:
            filename, filters = QFileDialog.getSaveFileName(self, caption='Save file', dir='.')
        else:
            msg_dialog = QMessageBox()
            msg_dialog.setIcon(QMessageBox.Information)
            msg_dialog.setText('No data to save.')
            msg_dialog.exec_()
            return

        if filename:
            self.parent.Data.export_matrices(filename)

    def quantify(self):
        if self.intercept.text().isdigit():
            intercept = int(self.intercept.text())
        else:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('Missing intercept.')
            error_dialog.exec_()
            return
        if self.slope.text().isdigit():
            slope = int(self.slope.text())
        else:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('Missing slope.')
            error_dialog.exec_()
            return

        elem = self.element.currentText()
        self.parent.Data.quantify_map(elem=elem, intercept=intercept, slope=slope)
        self.image(quantified=True)

    def quantify_all(self):
        msg_dialog = QMessageBox()
        msg_dialog.setIcon(QMessageBox.Information)
        msg_dialog.setText('Not supported yet.')
        msg_dialog.exec_()

    def export_quantified(self):
        if self.parent.Data.qmaps:
            filename, filters = QFileDialog.getSaveFileName(self, caption='Open file', dir='.')
        else:
            msg_dialog = QMessageBox()
            msg_dialog.setIcon(QMessageBox.Information)
            msg_dialog.setText('No data to save.')
            msg_dialog.exec_()
            return

        if filename:
            self.parent.Data.export_matrices(filename, quantified=True)
