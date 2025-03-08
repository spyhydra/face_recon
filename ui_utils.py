import tkinter as tk
from tkinter import ttk, messagebox
import os
import threading
import time
from PIL import Image, ImageTk
import config

class ProgressBar:
    """A progress bar widget for long-running operations"""
    
    def __init__(self, parent, title="Processing...", determinate=False):
        self.parent = parent
        self.title = title
        self.determinate = determinate
        self.popup = None
        self.progress = None
        self.label = None
        self.stop_event = threading.Event()
    
    def start(self, max_value=100):
        """Start the progress bar"""
        self.popup = tk.Toplevel(self.parent)
        self.popup.title(self.title)
        self.popup.geometry("300x100")
        self.popup.resizable(False, False)
        self.popup.configure(background=config.UI_THEME["bg_color"])
        self.popup.grab_set()  # Make the dialog modal
        
        # Center the popup
        self.popup.update_idletasks()
        width = self.popup.winfo_width()
        height = self.popup.winfo_height()
        x = (self.popup.winfo_screenwidth() // 2) - (width // 2)
        y = (self.popup.winfo_screenheight() // 2) - (height // 2)
        self.popup.geometry(f"{width}x{height}+{x}+{y}")
        
        # Add a label
        self.label = tk.Label(
            self.popup,
            text=self.title,
            bg=config.UI_THEME["bg_color"],
            fg=config.UI_THEME["fg_color"],
            font=("Verdana", 10)
        )
        self.label.pack(pady=10)
        
        # Add a progress bar
        self.progress = ttk.Progressbar(
            self.popup,
            orient=tk.HORIZONTAL,
            length=250,
            mode="determinate" if self.determinate else "indeterminate"
        )
        self.progress.pack(pady=10)
        
        if not self.determinate:
            self.progress.start()
        else:
            self.progress["maximum"] = max_value
            self.progress["value"] = 0
    
    def update(self, value=None, text=None):
        """Update the progress bar"""
        if self.popup is None:
            return
        
        if text is not None:
            self.label.config(text=text)
        
        if self.determinate and value is not None:
            self.progress["value"] = value
        
        self.popup.update()
    
    def stop(self):
        """Stop and close the progress bar"""
        if self.popup is None:
            return
        
        if not self.determinate:
            self.progress.stop()
        
        self.popup.grab_release()
        self.popup.destroy()
        self.popup = None

def create_rounded_button(parent, text, command, width=20, height=2, radius=10):
    """Create a rounded button with hover effect"""
    frame = tk.Frame(parent, bg=config.UI_THEME["bg_color"])
    
    def on_enter(e):
        button['background'] = config.UI_THEME["highlight_bg"]
        
    def on_leave(e):
        button['background'] = config.UI_THEME["button_bg"]
    
    button = tk.Button(
        frame,
        text=text,
        command=command,
        width=width,
        height=height,
        bg=config.UI_THEME["button_bg"],
        fg=config.UI_THEME["button_fg"],
        font=("Verdana", 10, "bold"),
        bd=0,
        activebackground=config.UI_THEME["highlight_bg"],
        activeforeground=config.UI_THEME["button_fg"],
        relief=tk.FLAT
    )
    button.pack(padx=10, pady=10)
    
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)
    
    return frame

def show_message(parent, title, message, message_type="info"):
    """Show a message dialog"""
    if message_type == "info":
        messagebox.showinfo(title, message, parent=parent)
    elif message_type == "warning":
        messagebox.showwarning(title, message, parent=parent)
    elif message_type == "error":
        messagebox.showerror(title, message, parent=parent)
    else:
        messagebox.showinfo(title, message, parent=parent)

def load_and_resize_image(image_path, width, height):
    """Load and resize an image"""
    try:
        img = Image.open(image_path)
        img = img.resize((width, height), Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def create_entry_with_label(parent, label_text, validate_cmd=None, width=20):
    """Create an entry field with a label"""
    frame = tk.Frame(parent, bg=config.UI_THEME["bg_color"])
    
    label = tk.Label(
        frame,
        text=label_text,
        bg=config.UI_THEME["bg_color"],
        fg=config.UI_THEME["fg_color"],
        font=("Verdana", 10)
    )
    label.pack(side=tk.LEFT, padx=5)
    
    if validate_cmd:
        entry = tk.Entry(
            frame,
            width=width,
            validate="key",
            validatecommand=validate_cmd
        )
    else:
        entry = tk.Entry(frame, width=width)
    
    entry.pack(side=tk.LEFT, padx=5)
    
    return frame, entry

def run_with_progress(parent, task_func, title="Processing...", args=(), kwargs={}):
    """Run a task with a progress bar"""
    progress = ProgressBar(parent, title)
    result = [None]  # Use a list to store the result
    
    def task():
        try:
            result[0] = task_func(*args, **kwargs)
        finally:
            parent.after(100, progress.stop)
    
    progress.start()
    threading.Thread(target=task, daemon=True).start()
    
    return result[0] 