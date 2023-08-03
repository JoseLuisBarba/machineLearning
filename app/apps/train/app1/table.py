import sys
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QTableWidget, QTableView, QTableWidgetItem

class Table(QTableWidget):
    colors = [
        ("Red", "#FF0000"),
        ("Green", "#00FF00"),
        ("Blue", "#0000FF"),
        ("Black", "#000000"),
        ("White", "#FFFFFF"),
        ("Electric Green", "#41CD52"),
        ("Dark Blue", "#222840"),
        ("Yellow", "#F9E56d")
    ]
    names = [
        'Name', 
        'Hex Code', 
        'Color'
    ]


    def __init__(self, parent= None) -> None:
        super(Table, self).__init__(parent)
        self.setRowCount(len(Table.colors))
        self.setColumnCount(len(Table.colors[0]) + 1)
        self.setHorizontalHeaderLabels(Table.names)
        self.displayTable()

    def displayTable(self,):
        for i, (name, code) in enumerate(Table.colors):

            itemName = QTableWidgetItem(name)

            itemCode = QTableWidgetItem(code)

            itemColor = QTableWidgetItem()
            itemColor.setBackground(self.getRGBFromHex(code))

            #add position
            self.setItem(i, 0, itemName)
            self.setItem(i, 1, itemCode)
            self.setItem(i, 2, itemColor)


        



    def getRGBFromHex(self, code: str):
        codeHex = code.replace('#','')
        rgb = tuple(int(codeHex[i:i+2],16) for i in (0, 2, 4))
        return QColor.fromRgb(rgb[0], rgb[1], rgb[2])
    



if __name__ == '__main__':
     #create the QT application
     app = QApplication(sys.argv)
     #create and show the form
     table = Table()
     table.show()

     # Run the main Qt loop
     sys.exit(app.exec())