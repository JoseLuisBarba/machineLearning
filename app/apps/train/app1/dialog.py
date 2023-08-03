import sys
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QPushButton, QVBoxLayout

class Form(QDialog):
    def __init__(self, parent= None) -> None:
        super(Form, self).__init__(parent)
        self.setWindowTitle('My Form')
        #create widgets
        self.edit = QLineEdit('Write my name here...')
        self.button = QPushButton('Show Greetings')
        # Create Layout and add widgets
        layout = QVBoxLayout(self)
        layout.addWidget(self.edit)
        layout.addWidget(self.button)
        #add button signal to greetings slot
        self.button.clicked.connect(self.greetings)
    #greets the user
    def greetings(self,):
        print(f'Hello {self.edit.text()}')



if __name__ == '__main__':
     #create the QT application
     app = QApplication(sys.argv)
     #create and show the form
     form = Form()
     form.show()

     # Run the main Qt loop
     sys.exit(app.exec())
