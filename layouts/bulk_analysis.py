import sys
from PySide2.QtWidgets import *
from PySide2 import QtGui
from PySide2.QtCore import Qt
from widgets.tab_table import TabTable
from widgets.pandas_model import PandasModel


class BulkAnalysis(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.mainWidget = QWidget()
        self.mainLayout = QVBoxLayout(self.mainWidget)
        self.mainLayout.setAlignment(Qt.AlignCenter)

        self.settingsGroup = QGroupBox('Settings')
        layout = QGridLayout()

        self.lodLbl = QLabel('Detection limit')
        layout.addWidget(self.lodLbl, 0, 0)
        self.lodCombo = QComboBox()
        self.lodCombo.addItems(['beginning', 'all'])
        layout.addWidget(self.lodCombo, 1, 0)

        self.standardLbl = QLabel('Standard')
        layout.addWidget(self.standardLbl, 0, 1)
        self.standard = QComboBox()
        self.standard.addItems(['Select standard'])
        layout.addWidget(self.standard, 1, 1)

        self.averageBtn = QPushButton('Average')
        layout.addWidget(self.averageBtn, 0, 2)
        self.averageBtn.clicked.connect(self.average)
        self.quantifyBtn = QPushButton('Quantification')
        layout.addWidget(self.quantifyBtn, 1, 2)
        self.quantifyBtn.clicked.connect(self.quantify)

        self.intStdCor = QPushButton('Internal Std Correction')
        layout.addWidget(self.intStdCor, 0, 3)
        self.intStdCor.clicked.connect(self.correctionIS)
        self.totSumCor = QPushButton('Total Sum Correction')
        layout.addWidget(self.totSumCor, 1, 3)
        self.totSumCor.clicked.connect(self.correctionTS)

        self.reportBtn = QPushButton('Report')
        layout.addWidget(self.reportBtn, 0, 4)
        self.reportBtn.clicked.connect(self.report)
        self.exportBtn = QPushButton('Export')
        layout.addWidget(self.exportBtn, 1, 4)
        self.exportBtn.clicked.connect(self.export)

        layout.setAlignment(Qt.AlignLeft)
        self.settingsGroup.setLayout(layout)

        self.mainLayout.addWidget(self.settingsGroup)

        self.table = TabTable(self)
        self.mainLayout.addWidget(self.table)

    def get_page(self):
        """
        Returns Page for bulk analysis for main stack of gui
        Returns
        -------
        mainWidget : QWidget
            The widget that the houses the bulk analysis screen.
        """
        return self.mainWidget

    def average(self):
        """
        Calculate average of peaks for MSData
        """
        method = self.parent.data_select.method.currentText()
        scale = self.lodCombo.currentText()
        self.parent.Data.average(method=method, scale=scale)

        if len(self.parent.Data.names) != len(self.parent.Data.average_peaks.index):
            error_dialog = QErrorMessage()
            error_dialog.showMessage('Number of peaks doesnt match number of given names.')
            error_dialog.exec_()
            return

        model = PandasModel(self.parent.Data.average_peaks)
        self.table.table['Average'].setModel(model)

    def quantify(self):
        """
        Calculate quantified values of peaks for MSData
        """
        method = self.parent.data_select.method.currentText()
        std = self.standard.currentText()
        try:
            if std not in self.parent.Data.names:
                error_dialog = QErrorMessage()
                error_dialog.showMessage('{} not in peak names.'.format(std))
                error_dialog.exec_()
                return
        except:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('Unexpected error.')
            error_dialog.exec_()
            return

        lod = self.lodCombo.currentText()
        self.parent.Data.set_srm(std)
        self.parent.Data.quantification()
        self.parent.Data.detection_limit(method=method, scale=lod)

        model = PandasModel(self.parent.Data.quantified)
        self.table.table['Quantified'].setModel(model)

    def correctionIS(self):
        """
        Correct quantified values of MSData using internal standard correction
        """
        if self.parent.Data.internal_std is None:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('Missing data for internal standard correction.')
            error_dialog.exec_()
            return

        self.parent.Data.internal_standard_correction()

        model = PandasModel(self.parent.Data.corrected_IS[0])
        self.table.table['Internal Std'].setModel(model)

    def correctionTS(self):
        """
        Correct quantified values of MSData using total sum correction
        """
        if self.parent.Data.sum_koeficients is None:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('Missing data for total sum correction.')
            error_dialog.exec_()
            return
        if self.parent.Data.quantified is None:
            self.error_msg('Missing data: Quantified.')
        try:
            self.parent.Data.total_sum_correction()
        except TypeError as e:
            self.error_msg('Unsuported operation. <br>{}'.format(e))

        model = PandasModel(self.parent.Data.corrected_SO)
        self.table.table['Total Sum'].setModel(model)

    def report(self):
        self.parent.Data.report()

        if self.parent.Data.quantified is not None:
            model = PandasModel(self.parent.Data.quantified)
            self.table.table['Quantified'].setModel(model)

        if self.parent.Data.corrected_IS is not None:
            model = PandasModel(self.parent.Data.corrected_IS[0])
            self.table.table['Internal Std'].setModel(model)

        if self.parent.Data.corrected_SO is not None:
            model = PandasModel(self.parent.Data.corrected_SO)
            self.table.table['Total Sum'].setModel(model)

    def export(self):
        data_to_save = self.table.return_tab()
        filename, filters = QFileDialog.getSaveFileName(self, 'Save file', '', 'All files (*.*);;Excel (*.xlsx, *.xls)')
        if filename:
            if data_to_save == 'Average':
                self.parent.Data.save(filename, self.parent.Data.average_peaks)
                self.missing_data(self.parent.Data.average_peaks, 'Average')
            elif data_to_save == 'Quantified':
                self.parent.Data.save(filename, self.parent.Data.quantified)
                self.missing_data(self.parent.Data.quantified, 'Quantified')
            elif data_to_save == 'Internal Std':
                self.parent.Data.save(filename, self.parent.Data.corrected_IS)
                self.missing_data(self.parent.Data.corrected_IS, 'Internal Std')
            elif data_to_save == 'Total Sum':
                self.parent.Data.save(filename, self.parent.Data.corrected_SO)
                self.missing_data(self.parent.Data.corrected_SO, 'Total Sum')
            else:
                self.missing_data(data_name='Unexpected')

    @staticmethod
    def missing_data(data=None, data_name='Unexpected'):
        if data_name == 'Unexpected':
            error_dialog = QErrorMessage()
            error_dialog.showMessage('Unexpected')
            error_dialog.exec_()
            return
        if data is None:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('Missing data: {}'.format(data_name))
            error_dialog.exec_()
            return

    @staticmethod
    def error_msg(msg):
        error_dialog = QErrorMessage()
        error_dialog.showMessage(msg)
        error_dialog.exec_()


