import os
import time
from tkinter import Tk, Button, Label, Canvas, ROUND, FALSE
from PIL import ImageGrab, Image


class Drawing:
    """Class responsible for getting user input via a drawing interface"""

    def __init__(self, x, y, image_length):
        """
        :param x: horizontal position of canvas
        :param y: vertical position of canvas
        """

        self.root = Tk()
        self.root.title('Input')
        self.root.geometry("300x330+" + str(0) + "+" + str(0))
        self.root.attributes("-topmost", True)

        # Done button
        self.button = Button(self.root, text='Done', command=self.end_program)
        self.button.grid(row=0, column=2)

        self.label = Label(self.root,
                           text='Draw any one digit then click "Done" ->')
        self.label.grid(row=0, column=0)

        # Drawing canvas
        self.canvas = Canvas(self.root, bg='black', width=300, height=300)
        self.canvas.grid(row=1, columnspan=5)

        self.x_pos = x
        self.y_pos = y
        self.old_x = None
        self.old_y = None
        self.line_width = 20
        self.color = 'white'
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
        self.image_length = image_length

        self.root.mainloop()

    def paint(self, event):
        """
        Paint line on left click

        :param event: Left-click event
        """
        self.line_width = 15
        paint_color = '#%02x%02x%02x' % (240, 240, 240)
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                    width=self.line_width, fill=paint_color,
                                    capstyle=ROUND, smooth=FALSE)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        """
        :param event: Left-click release event
        """
        self.old_x, self.old_y = None, None

    def end_program(self):
        """On program close, PIL screenshot and then save as JPEG image"""
        scale_factor = 1.5
        title_height = 40
        x1 = self.root.winfo_rootx()*scale_factor
        y1 = self.root.winfo_rooty()*scale_factor + title_height
        x2 = x1 + self.canvas.winfo_width()*scale_factor
        y2 = y1 + self.canvas.winfo_height()*scale_factor
        ImageGrab.grab().crop((x1, y1, x2, y2))\
            .resize((self.image_length, self.image_length), Image.ANTIALIAS)\
            .save(os.getcwd().rsplit('\\', 1)[0] + "/drawIn/input.jpg")

        self.root.destroy()

        time.sleep(2)
