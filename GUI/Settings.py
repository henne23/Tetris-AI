import tkinter as tk
import sys


class Settings():
    
    def __init__(self, game):
        self.game = game
        self.root = tk.Tk()
        self.root.title("Settings")
        self.settings = [tk.IntVar(value=1), tk.IntVar(), tk.IntVar(), tk.IntVar(value=1)]
        graphics_button = tk.Checkbutton(self.root, text="Graphics", variable=self.settings[0])
        manual_button = tk.Checkbutton(self.root, text="Manual control", variable=self.settings[1])
        train_button = tk.Checkbutton(self.root, text="Train mode", variable=self.settings[2])
        darkmode_button = tk.Checkbutton(self.root, text="Darkmode", variable=self.settings[3])
        graphics_button.pack(side=tk.LEFT)
        manual_button.pack(side=tk.LEFT)
        train_button.pack(side=tk.LEFT)
        darkmode_button.pack(side=tk.LEFT)

        settings = tk.Button(self.root, text="OK", command=self.get_settings)
        settings.pack(side=tk.BOTTOM)
        self.root.mainloop()

    def on_closing(self, event=None):
        #self.get_settings()
        self.root.destroy()
        sys.exit(0)

    def get_settings(self):
        self.game.graphics, self.game.manual, self.game.train, self.game.darkmode = [var.get() for var in self.settings]
        self.root.destroy()