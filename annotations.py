# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:58:54 2021

@author: AlexandreN
"""

import os
from os import listdir
from os.path import join, isdir
from os import chdir, getcwd

from tkinter import *
from PIL import Image, ImageTk
import math
import pandas as pd

BASE_STEP = 15


class LabellingApp(Frame):
    def __init__(self, folder, master=None):
        super().__init__(master)
        self.master = master

        wd = getcwd()
        chdir(wd)
        path = os.getcwd()
        self.folder = folder
        self.my_path = path + "\\data\\images\\"
        self.dirs = [f for f in listdir(self.my_path) if isdir(join(self.my_path, self.folder))]

        self.df = []

        self.i = 1
        self.last_i = 1
        self.step = 0
        self.last_step = 0

        self.currently_new_seq = True
        self.video_seq = 0

        # setting up a tkinter canvas
        self.w = Canvas(self.master, width=512, height=512)
        self.w.pack()

        self.new_image()

    def add_image(self, folder, i):
        file = self.my_path + folder + '\\im_' + str(i) + '.jpg'

        try:
            original = Image.open(file)
        except Exception:
            self.save_res()
            self.master.destroy()
        else:
            original = original.resize((512, 512))  # resize image

            self.img = ImageTk.PhotoImage(master=self.master, image=original)
            self.w.create_image(0, 0, image=self.img, anchor="nw")

    def new_image(self):
        print("newimage", self.i, "\nseq :", self.video_seq)
        if hasattr(self, 'b_redo'):
            self.b_redo.destroy()

        if hasattr(self, 'text'):
            self.w.delete(self.text)

        self.last_i = self.i
        self.i += self.step

        self.add_image(self.folder, self.i)
        self.text = self.w.create_text(10, 10, text="Cliquer si même séquence vidéo",
                                       fill="black", font='Helvetica 15 bold',
                                       anchor='nw')

        self.b_new_seq = Button(self.master,
                                text="Nouvelle séquence vidéo ?",
                                command=self.new_seq)
        self.b_new_seq.pack()
        self.b_save_res = Button(self.master,
                                 text="Sauvegarder données labellisées ?",
                                 command=self.save_res)
        self.b_save_res.pack()
        self.w.bind("<Button 1>", self.suite_sequence_video)

    def new_seq(self):
        self.w.delete(self.text)

        self.b_new_seq.destroy()
        self.b_save_res.destroy()

        if self.step > 1:
            self.i = self.last_i
            self.step = math.trunc(self.step / 2)
            self.new_image()
        else:
            self.get_label()
            self.currently_new_seq = True

    def save_res(self):
        print("saving...")
        df = pd.DataFrame(self.df)
        f_name = "labels_" + self.folder + '.csv'
        df.to_csv('data/dfs/'+f_name)
        print("done")

    def suite_sequence_video(self, event):
        self.w.delete(self.text)

        self.w.unbind("<Button 1>")
        self.b_new_seq.destroy()
        self.b_save_res.destroy()
        self.get_label()

    def get_label(self):
        self.w.unbind("<Button 1>")
        self.last_step = self.step
        self.step = BASE_STEP

        self.b1 = Button(self.master, text="Pull Up", command=self.pull_up_lab)
        self.b1.pack()
        self.b2 = Button(self.master, text="Not Pull Up", command=self.not_pull_up_lab)
        self.b2.pack()

        self.b_redo = Button(self.master, text="relabel current image", command=self.redo_image_process)
        self.b_redo.pack()

    def pull_up_lab(self):
        self.b1.destroy()
        self.b2.destroy()
        self.label= 1
        print("la")
        self.text = self.w.create_text(10, 10, text="Point Chest",
                                       fill="black", font='Helvetica 15 bold',
                                       anchor='nw')
        self.w.bind("<Button 1>", self.get_chest)

    def not_pull_up_lab(self):
        self.b1.destroy()
        self.b2.destroy()
        self.label= 0
        print("not pull_up")
        self.x_chest = 0
        self.y_chest = 0
        self.x_bar = 0
        self.y_bar = 0

        self.b_validate = Button(self.master, text="Validate label", command=self.validate_label)
        self.b_validate.pack()

    def redo_image_process(self):
        self.b1.destroy()
        self.b2.destroy()
        if hasattr(self, 'b_validate'):
            self.b_validate.destroy()

        self.i = self.last_i
        self.step = self.last_step
        self.new_image()

    def add_cross(self, x, y, col="dark"):
        self.w.create_line(x - 5, y, x + 5, y, fill=col)
        self.w.create_line(x, y - 5, x, y + 5, fill=col)

    def get_chest(self, event):
        self.w.delete(self.text)
        self.text = self.w.create_text(10, 10, text="Point the middle of the bar",
                                       fill="black", font='Helvetica 15 bold',
                                       anchor='nw')
        self.x_chest = event.x
        self.y_chest = event.y
        self.add_cross(self.x_chest, self.y_chest, "green")
        print("chest : ", self.x_chest, self.y_chest)

        self.w.bind("<Button 1>", self.get_bar)

    def get_bar(self, event):
        self.w.delete(self.text)
        self.w.unbind("<Button 1>")

        self.x_bar = event.x
        self.y_bar = event.y
        self.add_cross(self.x_bar, self.y_bar, "red")
        print("bar : ", self.x_bar, self.y_bar)

        self.b_validate = Button(self.master, text="Validate label", command=self.validate_label)
        self.b_validate.pack()

    def validate_label(self):
        self.b_validate.destroy()

        self.coord_bar = [self.x_bar / self.w.winfo_width(), self.y_bar / self.w.winfo_width()]
        self.coord_chest = [self.x_chest / self.w.winfo_width(), self.y_chest / self.w.winfo_width()]

        # outputting x and y coords to console
        print(self.label, self.coord_bar, self.coord_chest)

        self.annotate()

        self.currently_new_seq = False
        # lessgo for a new image to annotate
        self.new_image()

    def annotate(self):
        if self.currently_new_seq:
            self.video_seq += 1

        self.df.append({'name': "im_" + str(self.i),
                        'folder': self.folder,
                        'video_seq': self.video_seq,
                        'pull_up': self.label,
                        'bar': self.coord_bar,
                        'chest': self.coord_chest, })


def main(folder):
    # Create our master object to the Application
    master = Tk()
    # Create our application object
    app = LabellingApp(folder, master=master)
    # Start the mainloop
    app.mainloop()


if __name__ == "__main__":
    folder = 'images2'
    main(folder)
