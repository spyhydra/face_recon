import tkinter as tk
from tkinter import *
import os
import pandas as pd
import datetime
import tkinter.ttk as ttk

# Import our utility modules
import config
import db_utils
import ui_utils

def show_attendance(subject, parent_window=None):
    """
    Show attendance for a subject
    
    Args:
        subject: Subject name
        parent_window: Parent window
    """
    # Check if subject is provided
    if not subject:
        if parent_window:
            ui_utils.show_message(
                parent_window,
                "Error",
                "Please enter a subject name.",
                "error"
            )
        return
    
    # Get today's date
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Get attendance data
    attendance_df = db_utils.get_attendance(subject, date)
    
    if attendance_df.empty:
        if parent_window:
            ui_utils.show_message(
                parent_window,
                "Info",
                f"No attendance records found for {subject} on {date}.",
                "info"
            )
        return
    
    # Create a new window to display attendance
    attendance_window = Toplevel(parent_window) if parent_window else Tk()
    attendance_window.title(f"Attendance for {subject} - {date}")
    attendance_window.geometry("800x600")
    attendance_window.configure(background=config.UI_THEME["bg_color"])
    
    # Create a frame for the title
    title_frame = tk.Frame(attendance_window, bg=config.UI_THEME["bg_color"])
    title_frame.pack(fill=X, padx=10, pady=10)
    
    # Add title
    title_label = tk.Label(
        title_frame,
        text=f"Attendance for {subject} - {date}",
        bg=config.UI_THEME["bg_color"],
        fg=config.UI_THEME["fg_color"],
        font=("Verdana", 16, "bold")
    )
    title_label.pack()
    
    # Create a frame for the attendance table
    table_frame = tk.Frame(attendance_window, bg=config.UI_THEME["bg_color"])
    table_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)
    
    # Configure the treeview style
    style = ttk.Style()
    style.configure(
        "Treeview",
        background=config.UI_THEME["bg_color"],
        foreground=config.UI_THEME["fg_color"],
        rowheight=25,
        fieldbackground=config.UI_THEME["bg_color"]
    )
    style.map(
        "Treeview",
        background=[("selected", config.UI_THEME["highlight_bg"])]
    )
    
    # Create the treeview
    tree = ttk.Treeview(table_frame)
    tree["columns"] = ("Enrollment", "Name", "Time")
    
    # Configure columns
    tree.column("#0", width=0, stretch=NO)
    tree.column("Enrollment", anchor=CENTER, width=150)
    tree.column("Name", anchor=W, width=300)
    tree.column("Time", anchor=CENTER, width=150)
    
    # Configure headings
    tree.heading("#0", text="", anchor=CENTER)
    tree.heading("Enrollment", text="Enrollment", anchor=CENTER)
    tree.heading("Name", text="Name", anchor=CENTER)
    tree.heading("Time", text="Time", anchor=CENTER)
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=RIGHT, fill=Y)
    
    # Add data to the treeview
    for i, row in attendance_df.iterrows():
        tree.insert(
            "",
            END,
            values=(row["Enrollment"], row["Name"], row["Time"])
        )
    
    # Pack the treeview
    tree.pack(fill=BOTH, expand=True)
    
    # Create a frame for buttons
    button_frame = tk.Frame(attendance_window, bg=config.UI_THEME["bg_color"])
    button_frame.pack(fill=X, padx=10, pady=10)
    
    # Add export button
    export_button = ui_utils.create_rounded_button(
        button_frame,
        "Export to Excel",
        lambda: export_to_excel(attendance_df, subject, date),
        width=15,
        height=1
    )
    export_button.pack(side=LEFT, padx=10)
    
    # Add close button
    close_button = ui_utils.create_rounded_button(
        button_frame,
        "Close",
        attendance_window.destroy,
        width=10,
        height=1
    )
    close_button.pack(side=RIGHT, padx=10)
    
    # Start the main loop if no parent window
    if not parent_window:
        attendance_window.mainloop()

def export_to_excel(df, subject, date):
    """
    Export attendance data to Excel
    
    Args:
        df: DataFrame with attendance data
        subject: Subject name
        date: Date string
    """
    try:
        # Create export directory if it doesn't exist
        export_dir = os.path.join(config.BASE_DIR, "Exports")
        os.makedirs(export_dir, exist_ok=True)
        
        # Create export file path
        export_file = os.path.join(
            export_dir,
            f"{subject}_{date}.xlsx"
        )
        
        # Export to Excel
        df.to_excel(export_file, index=False)
        
        # Show success message
        ui_utils.show_message(
            None,
            "Success",
            f"Attendance exported to {export_file}",
            "info"
        )
        
    except Exception as e:
        # Show error message
        ui_utils.show_message(
            None,
            "Error",
            f"Error exporting attendance: {str(e)}",
            "error"
        )
        print(f"Error exporting attendance: {e}")

# For testing
if __name__ == "__main__":
    show_attendance("Test")
