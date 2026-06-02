import PySide6.QtWidgets as widg
from PySide6.QtWidgets import QApplication, QWidget, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget
from PySide6.QtGui import QImage, QPixmap, QFont, QFontMetrics
from PySide6.QtCore import Qt, QUrl, Signal

import cv2

from astropy.io import fits
from astropy.visualization import (
    ImageNormalize,
    PercentileInterval,
    AsinhStretch
)

import numpy as np
import sys
import os

class ElideButton(widg.QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self._full_text = text
        self.setText(text)

    def resizeEvent(self, event):
        metrics = QFontMetrics(self.font())
        elided = metrics.elidedText(
            self._full_text,
            Qt.ElideMiddle,
            self.width() - 80  # small padding tweak
        )
        super().setText(elided)
        super().resizeEvent(event)

class input_container(widg.QFrame):

    open_image_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.paths = []
        self.files = []
        
        self.setStyleSheet("background-color: #121417")

        # widgets
        input_header = widg.QLabel("Drop capture folder here...")
        input_header.setStyleSheet("color: #E6EDF3;")

        # layout
        self.input_layout = widg.QVBoxLayout()
        self.input_layout.setAlignment(Qt.AlignTop)
        self.input_layout.setContentsMargins(10, 10, 10, 10)
        self.input_layout.setSpacing(8)

        self.input_layout.addWidget(input_header, alignment=Qt.AlignCenter)

        self.setLayout(self.input_layout)
        self.setAcceptDrops(True)
        
        self.adjustSize()
    
    def clear_layout(self):
        self.files.clear()
        self.paths.clear()

        while self.input_layout.count():
            item = self.input_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:

                widget.deleteLater()
        input_header = widg.QLabel("Drop capture folder here...")
        input_header.setStyleSheet("color: #E6EDF3;")
        self.input_layout.addWidget(input_header, alignment=Qt.AlignCenter)

        self.adjustSize()
        self.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        folders = [p for p in paths if os.path.isdir(p)]

        self.clear_layout()

        if folders:
            event.acceptProposedAction()

            for folder in folders:
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        self.files.append(file)

                        full_path = os.path.join(root, file)
                        self.paths.append(full_path)

                        if file.lower().endswith((".fits", ".fit", ".tiff", ".tif", ".jpeg", ".jpg", ".png")):
                            new_file_button = ElideButton(file)
                            new_file_button.setFixedHeight(30)
                            new_file_button.setMinimumWidth(360)
                            new_file_button.setStyleSheet("""
                                                            QPushButton {
                                                                background-color: #343B46;
                                                                color: #E6EDF3;
                                                                Text-align: left;
                                                                padding-left: 10px;
                                                            }
                                                            QPushButton:hover {
                                                                background-color: #454D59;
                                                            }
                                                            QPushButton:pressed {
                                                                background-color: #1E2227;
                                                            }
                                                        """)
                            
                            layout = widg.QHBoxLayout()
                            score_label = widg.QLabel("--/10")
                            score_label.setStyleSheet("""color: #E6EDF3;
                                                      background-color: #343B46;""")
                            layout.addWidget(score_label, alignment=Qt.AlignRight)

                            new_file_button.setLayout(layout)
                            
                            new_file_button.clicked.connect(lambda checked=False, path=full_path: self.change_image(path))

                            self.input_layout.addWidget(new_file_button, alignment=Qt.AlignLeft)
            self.adjustSize()
    
    def change_image(self, file_path):
        print(f"Opened {file_path}")

        self.open_image_signal.emit(file_path)
        
        #print(f"Files: {self.files}")
        #print(f"Paths: {self.paths}")
    
    def score_frames(self):
        for i in range(self.input_layout.count()):
            item = self.input_layout.itemAt(i)
            if not item:
                continue
            widget = item.widget()
            if not widget:
                continue
            inner_layout = widget.layout()
            if not inner_layout:
                continue
            score_item = inner_layout.itemAt(0)
            if not score_item:
                continue
            score_text = score_item.widget()
            if not score_text:
                continue

            score_text.setText("10/10")

class image_container(widg.QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 400)
        self.setStyleSheet("background-color: #121417")

        self.display = widg.QLabel()
        self.display.setFixedSize(400, 400)
        self.display.setAlignment(Qt.AlignCenter)

        layout = widg.QHBoxLayout()
        layout.addWidget(self.display, alignment=Qt.AlignCenter)
        self.setLayout(layout)
    
    def update_display(self, new_image_path, new_image):
        if new_image_path:
            pixmap = QPixmap(new_image_path)
            self.display.setPixmap(pixmap.scaled(
                self.display.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

        if new_image:
            pixmap = QPixmap.fromImage(new_image)
            self.display.setPixmap(pixmap.scaled(
                self.display.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

    def recieve_image(self, file_path):
        if file_path.lower().endswith((".jpeg", ".jpg", ".png", ".tiff", ".tif")):
            self.update_display(file_path, None)
            print("Loaded Image")
        elif file_path.lower().endswith((".fits", ".fit")): 
            # Stretch
            data = fits.getdata(file_path)

            data = np.nan_to_num(data).astype(np.float32)

            data -= np.percentile(data, 5)
            data = np.clip(data, 0, np.percentile(data, 99.5))

            data /= (data.max() + 1e-8)
            data = np.arcsinh(data * 1)
            data /= data.max()

            data = (data * 255).astype(np.uint8)

            h, w = data.shape

            image = QImage(
                data.data,
                w,
                h,
                w,
                QImage.Format_Grayscale8
            )

            self.update_display(None, image)
        else:
            print("ERROR: file not supported")

class basic_button(widg.QPushButton):
    def __init__(self, text, hsize, vsize, text_color, background_color):
        super().__init__()
        self.setText(text)
        self.setFixedSize(hsize, vsize)
        self.setStyleSheet(f"""color: #{text_color};
                           background-color: #{background_color};""")

class main_window(QWidget):
    def __init__(self):
        super().__init__()

        # ------------------------------------------------------
        # Input files region
        # ------------------------------------------------------
        
        # region
        scroll_area = widg.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedSize(400, 400)

        self.input_container = input_container()
        scroll_area.setWidget(self.input_container)

        self.input_region = scroll_area

        # ------------------------------------------------------
        # Image shower
        # ------------------------------------------------------

        self.image_container = image_container()
        self.image_region = self.image_container

        # Connect signal
        self.input_container.open_image_signal.connect(self.image_container.recieve_image)

        # ------------------------------------------------------
        # Filter
        # ------------------------------------------------------

        filter_region = QWidget()
        filter_region.setFixedSize(802, 100)
        filter_region.setStyleSheet("background-color: #1E2227;")

        # Score and Filter region ---------------------

        left_buttons = QWidget()
        left_buttons.setFixedSize(200, 100)
        left_button_layout = widg.QVBoxLayout()

        score_images_button = basic_button("Score", 160, 25, "E6EDF3", "343B46")
        score_images_button.clicked.connect(self.input_container.score_frames)

        filter_images_button = basic_button("Filter", 160, 25, "E6EDF3", "343B46")
        #filter_images_button.clicked.connect(self.input_container.score_frames) -- ignore for now

        left_button_layout.addWidget(score_images_button, alignment=Qt.AlignCenter)
        left_button_layout.addWidget(filter_images_button, alignment=Qt.AlignCenter)
        left_button_layout.setAlignment(Qt.AlignTop)
        left_button_layout.setSpacing(15)

        left_buttons.setLayout(left_button_layout)

        # Filter setting region

        filter_settings = QWidget()
        filter_settings.setFixedSize(400, 100)
        filter_settings_layout = widg.QGridLayout()

        filter_settings.setLayout(filter_settings_layout)

        # Download folder region

        download_region = QWidget()
        download_region.setFixedSize(200, 100)
        download_region_layout = widg.QVBoxLayout()

        download_button = basic_button("Download", 160, 25, "E6EDF3", "343B46")
        #download_button.clicked.connect(self.input_container.score_frames) -- ignore for now

        download_region_layout.addWidget(download_button, alignment=Qt.AlignCenter)
        download_region_layout.setAlignment(Qt.AlignTop)
        download_region_layout.setSpacing(15)

        download_region.setLayout(download_region_layout)
        
        # Main filter region layout --------------------

        filter_layout = widg.QGridLayout()
        filter_layout.addWidget(left_buttons, 0, 0)
        filter_layout.addWidget(filter_settings, 0, 1)
        filter_layout.addWidget(download_region, 0, 2, )

        filter_region.setLayout(filter_layout)

        self.filter_region = filter_region

        # ------------------------------------------------------
        # Main window layout
        # ------------------------------------------------------

        window_layout = widg.QGridLayout()
        window_layout.addWidget(self.input_region, 0, 0)
        window_layout.addWidget(self.image_region, 0, 1)
        window_layout.addWidget(self.filter_region, 1, 0, 1, 2)

        window_layout.setContentsMargins(0, 0, 0, 0)
        window_layout.setSpacing(2)

        self.setLayout(window_layout)

app = QApplication(sys.argv)
app.setFont(QFont("Inter", 12))
app.setStyle("Fusion")

window = main_window()
window.setFixedSize(802, 502)
window.setStyleSheet("background-color: #343B46")
window.show()

app.exec()
