import tkinter as tk
from tkinter import filedialog

import numpy as np
import pydicom
from PIL import Image, ImageFilter, ImageTk

from UNET.evalFinal import segmentImage


class ImageProcessor:
    def __init__(self, image_path):
        self.original_image_pixels = None
        self.dicom_width = None
        self.dicom_center = None
        self.image_path = image_path
        self.original_image = None
        self.filtered_image = None
        self.segmented_image = None
        self.dicom_data = None

    def apply_median_filter(self):
        self.filtered_image = self.original_image.filter(ImageFilter.MedianFilter())

    def apply_gaussian_filter(self):
        self.filtered_image = self.original_image.filter(ImageFilter.GaussianBlur())


class App:
    def __init__(self, master):
        self.segmented_image = None
        self.original_image = None
        self.dicom_data = None
        self.original_image_pixels = None
        self.filtered_image = None
        self.master = master
        master.title("Image Processing App")

        # Initialize windowing values
        self.window_center = tk.DoubleVar(value=0)
        self.window_width = tk.DoubleVar(value=0)

        self.image_processor = None
        self.original_image_label = None
        self.filtered_image_label = None

        self.create_menu_bar()

    def create_menu_bar(self):
        menu_bar = tk.Menu(self.master)

        # File Menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.import_image)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.master.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        # View Menu
        view_menu = tk.Menu(menu_bar, tearoff=0)
        view_menu.add_command(label="Show Original", command=self.show_original_image)
        view_menu.add_separator()
        view_menu.add_command(label="Apply Median Filter", command=self.apply_median_filter)
        view_menu.add_command(label="Apply Gaussian Filter", command=self.apply_gaussian_filter)
        view_menu.add_command(label="Segment", command=self.segment)
        view_menu.add_command(label="Show Metadata", command=self.show_metadata)
        menu_bar.add_cascade(label="View", menu=view_menu)

        self.master.config(menu=menu_bar)

    def import_image(self):
        file_path = filedialog.askopenfilename()
        self.load_image(file_path)
        self.display_images()

    def load_image(self, file_path):
        if file_path.endswith('.dcm'):
            dicom_image = pydicom.dcmread(file_path)
            pixel_data = dicom_image.pixel_array
            min_pixel_value = np.min(pixel_data)
            max_pixel_value = np.max(pixel_data)
            # Initialize windowing values
            self.window_center = (min_pixel_value + max_pixel_value) / 2
            self.window_width = max_pixel_value - min_pixel_value
            self.original_image_pixels = dicom_image
            self.original_image = Image.fromarray(dicom_image.pixel_array)
            self.dicom_data = dicom_image
        else:
            self.original_image = Image.open(self.image_path)

        # Add sliders for windowing
        windowing_frame = tk.Frame(self.master)
        windowing_frame.pack(side=tk.TOP)

        center_label = tk.Label(windowing_frame, text="Window Center:")
        center_label.pack(side=tk.LEFT)

        center_slider = tk.Scale(windowing_frame, from_=0, to=2550, variable=self.window_center,orient=tk.HORIZONTAL, length=200, command=lambda x: self.update_windowing(center_slider.get(), width_slider.get()))

        center_slider.set(self.window_center)
        center_slider.pack(side=tk.LEFT)

        width_label = tk.Label(windowing_frame, text="Window Width:")
        width_label.pack(side=tk.LEFT)

        width_slider = tk.Scale(windowing_frame, from_=0, to=2550, variable=self.window_width, orient=tk.HORIZONTAL, length=200, command=lambda x: self.update_windowing(center_slider.get(), width_slider.get()))

        width_slider.set(self.window_width)
        width_slider.pack(side=tk.LEFT)

        self.filtered_image = self.original_image

    def apply_median_filter(self):
        self.image_processor.apply_median_filter()
        self.display_images()

    def apply_gaussian_filter(self):
        self.image_processor.apply_gaussian_filter()
        self.display_images()

    def show_original_image(self):
        self.image_processor.filtered_image = self.image_processor.original_image
        self.display_images()

    # Define a function to apply windowing to the pixel data
    def apply_windowing(self, pixel_data, window_center, window_width):
        min_value = window_center - (window_width / 2)
        max_value = window_center + (window_width / 2)
        pixel_data = np.clip(pixel_data, min_value, max_value)
        pixel_data = (pixel_data - min_value) / (max_value - min_value)
        pixel_data = np.clip(pixel_data, 0, 1)
        return pixel_data

    # Define a function to update the windowing when the sliders are moved
    def update_windowing(self, center_value, width_value):
        if self.original_image_pixels:
            pixel_data = self.apply_windowing(self.original_image_pixels.pixel_array.astype(float), center_value, width_value)
            img = Image.fromarray(np.uint8(pixel_data * 255))
            self.filtered_image = img
            self.update_filtered_image()

    def segment(self):
        filtered_image = segmentImage(np.array(self.original_image))
        filtered_image = filtered_image.astype(np.uint8)
        filtered_image = Image.fromarray(filtered_image)
        height, width = self.original_image.size
        filtered_image = filtered_image.resize((width, height))
        self.filtered_image = filtered_image
        self.segmented_image = filtered_image
        self.display_images()

    def show_metadata(self):
        if self.image_processor.dicom_data:
            metadata = tk.Toplevel(self.master)
            metadata.title("DICOM Metadata")
            metadata_text = tk.Text(metadata)
            metadata_text.pack(expand=True, fill=tk.BOTH)
            metadata_text.insert(tk.END, str(self.image_processor.dicom_data))
        else:
            tk.messagebox.showerror("Error", "No DICOM metadata available")

    def display_images(self):
        if self.original_image_label is not None:
            self.original_image_label.destroy()

        if self.filtered_image_label is not None:
            self.filtered_image_label.destroy()

        original_image = self.original_image
        filtered_image = self.filtered_image

        original_photo = ImageTk.PhotoImage(original_image)
        filtered_photo = ImageTk.PhotoImage(filtered_image)

        self.original_image_label = tk.Label(self.master, image=original_photo)
        self.original_image_label.image = original_photo
        self.original_image_label.pack(side=tk.LEFT)

        self.filtered_image_label = tk.Label(self.master, image=filtered_photo)
        self.filtered_image_label.image = filtered_photo
        self.filtered_image_label.pack(side=tk.LEFT)

    def update_filtered_image(self):
        if self.filtered_image_label is not None:
            self.filtered_image_label.destroy()

        filtered_image = self.filtered_image
        filtered_photo = ImageTk.PhotoImage(filtered_image)

        self.filtered_image_label = tk.Label(self.master, image=filtered_photo)
        self.filtered_image_label.image = filtered_photo
        self.filtered_image_label.pack(side=tk.LEFT)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
