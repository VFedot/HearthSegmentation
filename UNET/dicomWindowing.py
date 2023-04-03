import tkinter as tk
from tkinter import filedialog
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import filedialog
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DicomViewer:
    def __init__(self, master):
        self.master = master
        self.master.title("DICOM Viewer")

        # Create main frame
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas for displaying image
        self.figure = plt.figure(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.ax = self.figure.add_subplot(111)

        # Create buttons for opening DICOM file
        self.open_button = tk.Button(self.main_frame, text="Open DICOM", command=self.open_dicom)
        self.open_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Create sliders for adjusting window and level
        self.window_slider = tk.Scale(self.main_frame, from_=0, to=255, orient=tk.HORIZONTAL, length=200,
                                      label="Window", command=self.adjust_window)
        self.window_slider.pack(side=tk.LEFT, padx=10, pady=10)
        self.window_slider.set(128)
        self.level_slider = tk.Scale(self.main_frame, from_=0, to=255, orient=tk.HORIZONTAL, length=200, label="Level",
                                     command=self.adjust_level)
        self.level_slider.pack(side=tk.LEFT, padx=10, pady=10)
        self.level_slider.set(128)

        # Bind arrow keys to sliders
        self.master.bind("<Up>", self.step_up)
        self.master.bind("<Down>", self.step_down)

    def open_dicom(self):
        # Open file dialog to select DICOM file
        file_path = filedialog.askopenfilename(initialdir="./", title="Select DICOM file",
                                               filetypes=(("DICOM files", "*.dcm"),))
        if not file_path:
            return

        # Load DICOM file and display image
        dicom = pydicom.dcmread(file_path)
        image = dicom.pixel_array.astype(float)
        self.ax.imshow(image, cmap=plt.cm.gray)
        self.canvas.draw()

        # Set default window and level
        self.window = dicom.WindowWidth
        self.level = dicom.WindowCenter
        self.window_slider.config(from_=self.level, to=self.level + self.window)
        self.window_slider.set(self.window)
        self.level_slider.set(self.level)

    def adjust_window(self, value):
        self.window = int(value)
        self.ax.images[0].set_clim(self.level - self.window / 2, self.level + self.window / 2)
        self.canvas.draw()

    def adjust_level(self, value):
        self.level = int(value)
        self.ax.images[0].set_clim(self.level - self.window / 2, self.level + self.window / 2)
        self.canvas.draw()

    def step_up(self, event):
        if event.widget == self.window_slider:
            self.window_slider.set(self.window_slider.get() + 1)
        elif event.widget == self.level_slider:
            self.level_slider.set(self.level_slider.get() + 1)

    def step_down(self, event):
        if event.widget == self.window_slider:
            self.window_slider.set(self.window_slider.get() - 1)
        elif event.widget == self.level:
            self.level_slider.set(self.level_slider.get() - 1)


if __name__ == "__main__":
    root = tk.Tk()
    app = DicomViewer(root)
    root.mainloop()
