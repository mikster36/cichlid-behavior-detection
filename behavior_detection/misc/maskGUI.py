import tkinter as tk

import PIL.Image
import cv2
import numpy as np
from PIL import ImageTk
from PIL.Image import Image


class MaskSelector:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.image = None
        self.rect = None
        self.start_x, self.start_y = None, None

        self.ok_button = tk.Button(root, text="OK", command=self.get_rect)
        self.ok_button.pack()
        self.reset_button = tk.Button(root, text="Reset", command=self.reset_rect)
        self.reset_button.pack()

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.xy = None
        self.dimensions = None

    def open_image(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Could not open the video.")
            return 1

        ret, frame = cap.read()

        if ret:
            self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.image = PIL.Image.fromarray(self.image)
            self.image = self.image.resize((int(self.image.width / 2), int(self.image.height / 2)), PIL.Image.Resampling.NEAREST)
            self.img_tk = ImageTk.PhotoImage(image=self.image)

            self.canvas.config(width=self.image.width, height=self.image.height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
            self.reset_rect()

        cap.release()
        return 0

    def on_press(self, event):
        self.start_x, self.start_y = event.x, event.y

    def on_drag(self, event):
        if self.rect:
            self.canvas.delete(self.rect)
        cur_x, cur_y = event.x, event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, cur_x, cur_y, outline="white")

    def on_release(self, event):
        pass

    def reset_rect(self):
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None
            self.start_x, self.start_y = None, None

    def get_rect(self):
        if self.rect:
            bbox = self.canvas.coords(self.rect)
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            self.xy = (2 * x1, 2 * y1)
            self.dimensions = (2 * width, 2 * height)
            self.root.destroy()


def get_mask(video: str):
    root = tk.Tk()
    root.title("Mask Selector")
    mask = MaskSelector(root)
    if mask.open_image(video) == 1:
        x = int(input("X:" ))
        y = int(input("Y:" ))
        w = int(input("Width:" ))
        l = int(input("Length: "))
        root.destroy()
        return (x, y), (w, l)
    root.mainloop()
    return mask.xy, mask.dimensions

