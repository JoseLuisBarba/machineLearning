import sys
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import ( 
    QApplication, 
    QLabel, 
    QWidget, 
    QListWidget, 
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
)


class Widget(QWidget):
    def __init__(self, parent = None) -> None:

        super(Widget, self).__init__(parent)

        menuWidget = QListWidget()

        for i in range(10):
            item = QListWidgetItem(f'Item {i}')
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            menuWidget.addItem(item)

        textWidget = QLabel()
        button = QPushButton("Something")

        contentLayout = QVBoxLayout()

        contentLayout.addWidget(textWidget)
        contentLayout.addWidget(button)

        mainWidget = QWidget()
        mainWidget.setLayout(contentLayout)


        layout = QHBoxLayout()
        layout.addWidget(menuWidget, 1)
        layout.addWidget(mainWidget, 4)

        self.setLayout(layout)







if __name__ == '__main__':
    app = QApplication()

    w = Widget()
    w.show()
    fileStyle = 'C://Users//Admin//Documents//AdvancedAI//app//apps//train//app1//style.qss'
    with open(fileStyle, "r") as f:
        _style = f.read()
        app.setStyleSheet(_style)

    sys.exit(app.exec())