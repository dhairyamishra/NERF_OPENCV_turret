import cv2
import numpy as np
import logging
import datetime
import os
import threading
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
import tkinter as tk
from tkinter import ttk

# === SETUP ===
console = Console()
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_filename = log_dir / f"log_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting Computer Vision Dashboard")

# === DETECTION MODES ===
DETECTION_MODES = ["face", "fingers", "shapes"]
current_mode = DETECTION_MODES[0]

# === LOAD MODELS ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect(frame, mode):
    if mode == "face":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return frame, gray
    elif mode == "fingers":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([20, 150, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_frame = frame.copy()
        cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)
        return contour_frame, mask
    elif mode == "shapes":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_frame = frame.copy()
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            if len(approx) == 3:
                label = "Triangle"
            elif len(approx) == 4:
                label = "Rectangle"
            elif len(approx) > 10:
                label = "Circle"
            else:
                label = "Polygon"
            x, y = approx[0][0]
            cv2.putText(contour_frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return contour_frame, thresh
    else:
        return frame, np.zeros_like(frame)

# === GUI THREAD ===
class Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Computer Vision Dashboard")
        self.mode_var = tk.StringVar(value=current_mode)

        tk.Label(root, text="Detection Mode").pack()
        self.mode_menu = ttk.Combobox(root, values=DETECTION_MODES, textvariable=self.mode_var)
        self.mode_menu.pack()

        # Start CV processing in separate thread
        self.cv_thread = threading.Thread(target=self.run_cv)
        self.cv_thread.daemon = True
        self.cv_thread.start()

    def get_mode(self):
        return self.mode_var.get()

    def run_cv(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to grab frame")
                break

            mode = self.get_mode()
            try:
                processed, mask_or_gray = detect(frame.copy(), mode)
            except Exception as e:
                logging.exception("Error in detection")
                processed, mask_or_gray = frame.copy(), np.zeros_like(frame)

            cv2.imshow("1: Raw Feed", frame)
            cv2.imshow("2: Mask/Contours", mask_or_gray)
            cv2.imshow("3: Inference", processed)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

# === MAIN LOOP ===
def run_cv():
    cap = cv2.VideoCapture(0)
    dashboard = Dashboard()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to grab frame")
            break

        mode = dashboard.get_mode()
        try:
            processed, mask_or_gray = detect(frame.copy(), mode)
        except Exception as e:
            logging.exception("Error in detection")
            processed, mask_or_gray = frame.copy(), np.zeros_like(frame)

        cv2.imshow("1: Raw Feed", frame)
        cv2.imshow("2: Mask/Contours", mask_or_gray)
        cv2.imshow("3: Inference", processed)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

# === TERMINAL DISPLAY ===
def start_terminal_ui():
    table = Table(title="CV Dashboard Status")
    table.add_column("Key", justify="right", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Run Log", str(log_filename))
    table.add_row("Detection Modes", ", ".join(DETECTION_MODES))
    table.add_row("Exit", "Press ESC in any video window")

    panel = Panel(table, title="Welcome", subtitle="Interactive CV Dashboard")

    with Live(panel, refresh_per_second=1):
        root = tk.Tk()
        app = Dashboard(root)
        root.mainloop()

start_terminal_ui()
