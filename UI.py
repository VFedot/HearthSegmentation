import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageFilter, ImageTk
import pydicom


class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = None
        self.filtered_image = None

        self.load_image()

    def load_image(self):
        if self.image_path.endswith('.dcm'):
            dicom_image = pydicom.dcmread(self.image_path)
            self.original_image = Image.fromarray(dicom_image.pixel_array)
        else:
            self.original_image = Image.open(self.image_path)

        self.filtered_image = self.original_image

    def apply_median_filter(self):
        self.filtered_image = self.original_image.filter(ImageFilter.MedianFilter())

    def apply_gaussian_filter(self):
        self.filtered_image = self.original_image.filter(ImageFilter.GaussianBlur())


class App:
    def __init__(self, master):
        self.master = master
        master.title("Image Processing App")

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
        menu_bar.add_cascade(label="View", menu=view_menu)

        self.master.config(menu=menu_bar)


    def import_image(self):
        file_path = filedialog.askopenfilename()
        self.image_processor = ImageProcessor(file_path)
        self.display_images()

    def apply_median_filter(self):
        self.image_processor.apply_median_filter()
        self.display_images()

    def apply_gaussian_filter(self):
        self.image_processor.apply_gaussian_filter()
        self.display_images()

    def show_original_image(self):
        self.image_processor.filtered_image = self.image_processor.original_image
        self.display_images()

    def display_images(self):
        if self.original_image_label is not None:
            self.original_image_label.destroy()

        if self.filtered_image_label is not None:
            self.filtered_image_label.destroy()

        original_image = self.image_processor.original_image
        filtered_image = self.image_processor.filtered_image

        original_photo = ImageTk.PhotoImage(original_image)
        filtered_photo = ImageTk.PhotoImage(filtered_image)

        self.original_image_label = tk.Label(self.master, image=original_photo)
        self.original_image_label.image = original_photo
        self.original_image_label.pack(side=tk.LEFT)

        self.filtered_image_label = tk.Label(self.master, image=filtered_photo)
        self.filtered_image_label.image = filtered_photo
        self.filtered_image_label.pack(side=tk.LEFT)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
