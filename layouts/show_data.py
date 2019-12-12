from PySide2.QtWidgets import *
from PySide2.QtCore import Qt
from widgets.pandas_model import PandasModel


class ShowData(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.mainWidget = QWidget()
        self.mainLayout = QVBoxLayout(self.mainWidget)
        self.mainLayout.setAlignment(Qt.AlignCenter)

        self.group = QGroupBox('')
        layout = QHBoxLayout()
        layout.addStretch()
        self.elem = QComboBox()
        self.elem.currentTextChanged.connect(self.show_data)
        layout.addWidget(self.elem)
        layout.addStretch()
        self.group.setLayout(layout)

        self.mainLayout.addWidget(self.group)
        self.table = QTableView()
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

    def show_data(self):
        if self.parent.Data.maps:
            elem = self.elem.currentText()
            model = PandasModel(self.parent.Data.maps[elem])
            self.table.setModel(model)
