import tkinter as tk


class Settings():
    
    def __init__(self, game):
        self.game = game
        self.root = tk.Tk()
        self.root.title("Settings")
        self.settings = [tk.IntVar(), tk.IntVar(), tk.IntVar()]
        graphics_button = tk.Checkbutton(self.root, text="Graphics", variable=self.settings[0])
        manual_button = tk.Checkbutton(self.root, text="Manual control", variable=self.settings[1])
        train_button = tk.Checkbutton(self.root, text="Train mode", variable=self.settings[2])
        graphics_button.pack(side=tk.LEFT)
        manual_button.pack(side=tk.LEFT)
        train_button.pack(side=tk.LEFT)

        settings = tk.Button(self.root, text="OK", command=self.get_settings)
        settings.pack(side=tk.BOTTOM)
        self.root.mainloop()

    def get_settings(self):
        self.game.graphics, self.game.manual, self.game.train = [var.get() for var in self.settings]
        self.root.destroy()