import sys
from PySide6.QtAxContainer import QAxSelect, QAxWidget
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QMainWindow,
    QMessageBox,
    QToolBar
)


class MainWindow(QMainWindow):
    def __init__(self,) -> None:
        super().__init__()

        #toolBar
        toolBar = QToolBar()
        self.addToolBar(toolBar)

        #MenuBar
        fileMenu = self.menuBar().addMenu('&File')
        aboutMenu = self.menuBar().addMenu('&About')

        #action 1
        loadAction = QAction("Load...",self)
        loadAction.shortcut = 'Ctrl+L'
        loadAction.triggered = self.load
        fileMenu.addAction(loadAction)
        toolBar.addAction(loadAction)

        #action2 
        exitAction = QAction("E&xit",self)
        exitAction.shortcut = 'Ctrl+Q'
        exitAction.triggered = self.close
        fileMenu.addAction(exitAction)

        #action3
        aboutQtAct =  QAction('About &Qt', self)
        aboutQtAct.shortcut = QApplication.aboutQt
        aboutMenu.addAction( aboutQtAct)

        self.axWidget = QAxWidget()
        self.setCentralWidget(self.axWidget)

    def load(self,):
        axSelect = QAxSelect(self)
        if axSelect.exec() == QDialog.Accepted:
            clsid = axSelect.clsid()
            if not self.axWidget.setControl(clsid):
                QMessageBox.warning(self, "AxViewer", f"Unable to load {clsid}.")
    
    def close(self,):
        pass



if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    availableGeometry = mainWin.screen().availableGeometry()
    mainWin.resize(availableGeometry.width() / 3, availableGeometry.height() / 2)
    mainWin.show()
    sys.exit(app.exec())
