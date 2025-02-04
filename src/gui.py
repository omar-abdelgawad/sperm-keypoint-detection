"""Module containing gui related features of the app."""

import os
import sys
from typing import Optional
from threading import Thread
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

from . import cfg


class CustomButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        # self.config(
        #     relief=tk.FLAT,  # Remove button relief
        #     bd=0,  # Remove border
        #     highlightthickness=0,  # Remove highlight
        #     padx=10,  # Add horizontal padding
        #     pady=5,  # Add vertical padding
        #     font=("Arial", 12),  # Set font
        #     foreground="white",  # Text color
        #     background="orange",  # Background color
        # )
        # Bind events
        self.original_color = self["background"]
        self.bind("<Enter>", self.on_hover)
        self.bind("<Leave>", self.on_leave)

    def on_hover(self, _):
        self.config(background="lightblue")  # Change color on hover

    def on_leave(self, _):
        self.config(background=self.original_color)  # Restore original color


class GUI:
    """GUI interface for the app. Will be used to get input from the user."""

    def __init__(self, main_function) -> None:
        print("initialized app")
        self.other_thread: Optional[Thread] = None
        self.main_function = main_function
        self.window = tk.Tk()
        self.window_bg_color = "#353839"
        self.window.configure(bg=self.window_bg_color)
        self.window.title("Sperm Analyzer")
        self.window.geometry("500x500")
        self.window.resizable(False, False)
        # self.window.wm_attributes("-transparentcolor", "red")
        self.window.bind("<Destroy>", self.closing_procedure)
        # argv to be passed later
        self.mag = tk.StringVar()
        self.input_path = tk.StringVar()
        self.sampling_rate = tk.StringVar()
        # first row
        self.input_path_button = CustomButton(
            self.window, text="Browse video", command=self.open_video
        )
        self.magnification_combobox = ttk.Combobox(
            self.window,
            width=10,
            textvariable=self.mag,
            values=cfg.MAGNIFICATION_LIST,
            state="readonly",
        )
        # create an entry for writing sampling rate as an integer
        self.sampling_rate_entry = tk.Entry(
            self.window, textvariable=self.sampling_rate, width=10, justify="center"
        )
        # place them side by side but centered and with space between them without grid
        self.input_path_button.place(relx=0.2, rely=0.3, anchor=tk.CENTER)
        self.magnification_combobox.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
        self.sampling_rate_entry.place(relx=0.8, rely=0.3, anchor=tk.CENTER)
        # create three labels above them
        # make text bold
        self.input_path_label = tk.Label(
            self.window,
            text="Input Video Path",
            bg=self.window_bg_color,
            fg="white",
            font=("Helvetica", 10, "bold"),
        )
        self.input_path_label.place(relx=0.2, rely=0.2, anchor=tk.CENTER)
        self.magnification_label = tk.Label(
            self.window,
            text="Magnification",
            bg=self.window_bg_color,
            fg="white",
            font=("Helvetica", 10, "bold"),
        )
        self.magnification_label.place(relx=0.5, rely=0.2, anchor=tk.CENTER)
        self.sampling_rate_label = tk.Label(
            self.window,
            text="FPS",  # changed from sampling rate
            bg=self.window_bg_color,
            fg="white",
            font=("Helvetica", 10, "bold"),
        )
        self.sampling_rate_label.place(relx=0.8, rely=0.2, anchor=tk.CENTER)
        # create one more label for saved input path label
        self.saved_input_path_label = tk.Label(
            self.window, text="No file selected", width=20
        )
        self.saved_input_path_label.place(
            relx=0.5, rely=0.5, anchor=tk.CENTER, height=50
        )
        # logger label
        self.logger_label = tk.Label(self.window, text="Welcome", width=50)
        self.logger_label.place(relx=0.5, rely=0.8, anchor=tk.CENTER)
        # second row
        self.submit_button = CustomButton(
            self.window, text="Submit", command=self.submit
        )
        self.submit_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

    def closing_procedure(self, event) -> None:
        """Callback function for closing the window."""
        # self.other_thread.join() #waits for thread to finish before closing
        print("closed app", event, event.widget)
        sys.exit("Exiting app")

    def open_video(self) -> None:
        """Opens a file dialog to browse a video file."""
        self.input_path.set(
            filedialog.askopenfilename(
                filetypes=[
                    (
                        "Video Files",
                        " ".join([f"*{ext}" for ext in cfg.ALLOWED_VIDEO_EXTENSIONS]),
                    )
                ]
            )
        )
        if self.input_path.get():
            self.saved_input_path_label.configure(
                text=os.path.split(self.input_path.get())[1]
            )

    def submit(self) -> None:
        """Callback function for submit button."""
        print("submit button pressed")
        argv_dict = {
            "-i": self.input_path.get(),
            "-m": self.mag.get(),
            "-r": self.sampling_rate.get(),
        }
        items = [v for k, v in argv_dict.items()]
        for item in items:
            if not item:
                self.logger_label.configure(text="Please provide all fields")
                return
        # make argv from the dict that contains all keys and values as two elements
        # lists
        argv = [item for tup in argv_dict.items() for item in tup]
        print(argv)
        self.logger_label.configure(text="Task started")
        self.window.update()
        self.other_thread = Thread(target=self.main_function, args=(argv,))
        self.other_thread.start()
        self.logger_label.configure(text="Task Started")

    def run(self) -> None:
        """Run main app loop."""
        self.window.mainloop()
