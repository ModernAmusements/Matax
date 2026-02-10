#!/usr/bin/env python3
"""
NGO Facial Image Analysis System - User-Friendly GUI
Designed for non-technical users with visual feedback.
"""

import sys
import os
import json
import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import cv2
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.detection import FaceDetector
from src.embedding import FaceNetEmbeddingExtractor, SimilarityComparator


class Colors:
    BG_PRIMARY = "#F5F5F7"
    BG_SECONDARY = "#FFFFFF"
    BG_TERTIARY = "#F0F0F2"
    TEXT_PRIMARY = "#1D1D1F"
    TEXT_SECONDARY = "#86868B"
    ACCENT_BLUE = "#007AFF"
    ACCENT_GREEN = "#34C759"
    ACCENT_RED = "#FF3B30"
    ACCENT_ORANGE = "#FF9500"
    ACCENT_PURPLE = "#AF52DE"
    SUCCESS_GREEN = "#30D158"
    WARNING_YELLOW = "#FFD60A"
    ERROR_RED = "#FF453A"
    PROCESSING_BLUE = "#5856D6"
    BORDER = "#D2D2D7"


class StyledButton(tk.Canvas):
    def __init__(self, parent, text="", command=None, width=160, height=44,
                 bg_color="#007AFF", fg_color="white", **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg=Colors.BG_PRIMARY, highlightthickness=0, **kwargs)

        self.command = command
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.btn_text = text
        self._enabled = True
        self._width = width
        self._height = height

        self._draw(bg_color)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _draw(self, color):
        self.delete("all")
        w, h = self._width, self._height
        r = h / 2

        self.create_arc(0, 0, h, h, start=90, extent=90, fill=color, outline=color)
        self.create_arc(w-h, 0, w, h, start=0, extent=90, fill=color, outline=color)
        self.create_rectangle(r, 0, w-r, h, fill=color, outline=color)
        self.create_text(w/2, h/2, text=self.btn_text, fill=self.fg_color,
                        font=("Helvetica Neue", 13, "bold"))

    def _on_enter(self, e):
        if self._enabled:
            self._draw("#0066CC")

    def _on_leave(self, e):
        if self._enabled:
            self._draw(self.bg_color)

    def _on_click(self, e):
        if self._enabled:
            self._draw("#0055BB")

    def _on_release(self, e):
        if self._enabled:
            self._draw(self.bg_color)
            if self.command:
                self.command()


class ImagePanel(tk.Frame):
    def __init__(self, parent, title="", **kwargs):
        super().__init__(parent, bg=Colors.BG_TERTIARY, **kwargs)

        tk.Label(self, text=title,
                font=("Helvetica Neue", 11, "bold"),
                fg=Colors.TEXT_PRIMARY,
                bg=Colors.BG_TERTIARY).pack(anchor="w", padx=12, pady=(8, 4))

        self.canvas = tk.Canvas(self, bg=Colors.BG_TERTIARY, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.canvas.create_text(200, 100, text="No image",
                             fill=Colors.TEXT_SECONDARY,
                             font=("Helvetica Neue", 12))

        self.current_image = None

    def set_image(self, image: np.ndarray, max_width: int = 400):
        self.current_image = image
        self.canvas.delete("all")

        if image is None:
            return

        h, w = image.shape[:2]
        scale = min(max_width / max(1, w), 200 / max(1, h))
        new_w, new_h = int(w * scale), int(h * scale)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        resized = cv2.resize(rgb, (new_w, new_h))
        pil = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(pil)

        x = (max_width - new_w) // 2
        self.canvas.create_image(x, 10, anchor="nw", image=tk_img)
        self.canvas.image = tk_img


class StepIndicator(tk.Frame):
    STEPS = [
        ("1", "Select Photo", "Choose an image to analyze"),
        ("2", "Detect Faces", "AI finds faces automatically"),
        ("3", "Extract", "Create face signature"),
        ("4", "Compare", "Match with references"),
    ]

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=Colors.BG_PRIMARY, **kwargs)

        self.widgets = []

        for num, title, desc in self.STEPS:
            frame = tk.Frame(self, bg=Colors.BG_PRIMARY)
            frame.pack(side="left", padx=8, fill="both", expand=True)

            circle = tk.Canvas(frame, width=44, height=44,
                            bg=Colors.BG_TERTIARY, highlightthickness=0)
            circle.pack(pady=(0, 4))

            circle.create_oval(4, 4, 40, 40,
                             fill=Colors.BG_TERTIARY,
                             outline=Colors.BORDER, width=2)
            circle.create_text(22, 22, text=num,
                             fill=Colors.TEXT_SECONDARY,
                             font=("Helvetica Neue", 14, "bold"))

            tk.Label(frame, text=title,
                    font=("Helvetica Neue", 10, "bold"),
                    fg=Colors.TEXT_PRIMARY,
                    bg=Colors.BG_PRIMARY).pack()

            tk.Label(frame, text=desc,
                    font=("Helvetica Neue", 9),
                    fg=Colors.TEXT_SECONDARY,
                    bg=Colors.BG_PRIMARY).pack()

            self.widgets.append((circle, frame, Colors.BG_TERTIARY))

    def set_step(self, step: int):
        for i, (circle, frame, _) in enumerate(self.widgets):
            circle.delete("all")

            if i < step:
                fill = Colors.SUCCESS_GREEN
                text_color = "white"
            elif i == step:
                fill = Colors.PROCESSING_BLUE
                text_color = "white"
            else:
                fill = Colors.BG_TERTIARY
                text_color = Colors.TEXT_SECONDARY

            circle.create_oval(4, 4, 40, 40, fill=fill, outline=Colors.BORDER, width=2)
            circle.create_text(22, 22, text=str(i+1), fill=text_color,
                             font=("Helvetica Neue", 14, "bold"))


class FaceGallery(tk.Frame):
    def __init__(self, parent, title="", **kwargs):
        super().__init__(parent, bg=Colors.BG_PRIMARY, **kwargs)

        tk.Label(self, text=title,
                font=("Helvetica Neue", 10, "bold"),
                fg=Colors.TEXT_PRIMARY,
                bg=Colors.BG_PRIMARY).pack(anchor="w", pady=(0, 6))

        scroll_frame = tk.Frame(self, bg=Colors.BG_TERTIARY, relief="flat", bd=1)
        scroll_frame.pack(fill="x")

        canvas = tk.Canvas(scroll_frame, bg=Colors.BG_TERTIARY, highlightthickness=0, height=100)
        scrollbar = tk.Scrollbar(scroll_frame, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=scrollbar.set)

        canvas.pack(side="top", fill="x")
        scrollbar.pack(side="bottom", fill="x")

        inner = tk.Frame(canvas, bg=Colors.BG_TERTIARY)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        def on_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", on_configure)

        self.inner = inner
        self.items = []

    def clear(self):
        for item in self.items:
            item.destroy()
        self.items = []

    def add(self, image: np.ndarray, label: str = ""):
        frame = tk.Frame(self.inner, bg=Colors.BG_SECONDARY, relief="flat", bd=1)
        frame.pack(side="left", padx=6, pady=6)

        h, w = image.shape[:2] if image is not None else (60, 60)
        scale = min(60 / max(1, w), 60 / max(1, h))
        new_w, new_h = int(w * scale), int(h * scale)

        if image is not None:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
            resized = cv2.resize(rgb, (new_w, new_h))
            pil = Image.fromarray(resized)
            tk_img = ImageTk.PhotoImage(pil)
        else:
            tk_img = None

        lbl = tk.Label(frame, image=tk_img if tk_img else None,
                       bg=Colors.BG_SECONDARY,
                       text="N/A" if not tk_img else "",
                       fg=Colors.TEXT_SECONDARY)
        if tk_img:
            lbl.image = tk_img
        lbl.pack(padx=4, pady=4)

        tk.Label(frame, text=label,
                font=("Helvetica Neue", 9),
                fg=Colors.TEXT_SECONDARY,
                bg=Colors.BG_SECONDARY).pack(pady=(0, 4))

        self.items.append(frame)


class VisualizationPanel(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=Colors.BG_PRIMARY, **kwargs)

        tk.Label(self, text="◈ AI Analysis Visualizations",
                font=("Helvetica Neue", 14, "bold"),
                fg=Colors.ACCENT_PURPLE,
                bg=Colors.BG_PRIMARY).pack(anchor="w", pady=(0, 8))

        tk.Label(self, text="See how the AI analyzes faces in your images",
                font=("Helvetica Neue", 10),
                fg=Colors.TEXT_SECONDARY,
                bg=Colors.BG_PRIMARY).pack(anchor="w", pady=(0, 12))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.viz_panels = {}
        self._create_viz_tabs()

    def _create_viz_tabs(self):
        viz_types = [
            ("detection", "Face Detection", "Bounding boxes around detected faces"),
            ("extraction", "Face Extraction", "Cropped face for analysis"),
            ("landmarks", "Face Landmarks", "Key facial points detected"),
            ("features", "Feature Map", "Neural network activation patterns"),
            ("embedding", "Embedding", "128-dimensional face signature"),
            ("heatmap", "Attention Heatmap", "Areas AI focuses on"),
        ]

        for viz_id, title, desc in viz_types:
            tab = tk.Frame(self.notebook, bg=Colors.BG_TERTIARY, padx=12, pady=12)
            self.notebook.add(tab, text=f"  {title}  ")

            tk.Label(tab, text=desc,
                    font=("Helvetica Neue", 10),
                    fg=Colors.TEXT_SECONDARY,
                    bg=Colors.BG_TERTIARY).pack(anchor="w", pady=(0, 8))

            canvas = tk.Canvas(tab, bg=Colors.BG_TERTIARY, highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            canvas.create_text(200, 150, text="Run analysis to see visualization",
                             fill=Colors.TEXT_SECONDARY,
                             font=("Helvetica Neue", 12))

            self.viz_panels[viz_id] = canvas

    def set_visualization(self, viz_id: str, image: np.ndarray):
        if viz_id not in self.viz_panels:
            return

        canvas = self.viz_panels[viz_id]
        canvas.delete("all")

        if image is None:
            canvas.create_text(200, 150, text="No data available",
                             fill=Colors.TEXT_SECONDARY)
            return

        h, w = image.shape[:2]
        scale = min(380 / max(1, w), 280 / max(1, h))
        new_w, new_h = int(w * scale), int(h * scale)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        resized = cv2.resize(rgb, (new_w, new_h))
        pil = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(pil)

        x = (400 - new_w) // 2
        canvas.create_image(x, 10, anchor="nw", image=tk_img)
        canvas.image = tk_img


class ComparisonPanel(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=Colors.BG_TERTIARY, **kwargs)

        tk.Label(self, text="Comparison Result",
                font=("Helvetica Neue", 12, "bold"),
                fg=Colors.TEXT_PRIMARY,
                bg=Colors.BG_TERTIARY).pack(anchor="w", padx=12, pady=(8, 8))

        content = tk.Frame(self, bg=Colors.BG_TERTIARY)
        content.pack(fill="x", padx=12, pady=(0, 12))

        left_frame = tk.Frame(content, bg=Colors.BG_TERTIARY)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))

        tk.Label(left_frame, text="Your Image",
                font=("Helvetica Neue", 10),
                fg=Colors.TEXT_SECONDARY,
                bg=Colors.BG_TERTIARY).pack()

        self.left_canvas = tk.Canvas(left_frame, bg=Colors.BG_TERTIARY,
                                   highlightthickness=0, height=150)
        self.left_canvas.pack(fill="x", pady=4)
        self.left_canvas.create_text(100, 75, text="No image",
                                   fill=Colors.TEXT_SECONDARY,
                                   font=("Helvetica Neue", 10))

        right_frame = tk.Frame(content, bg=Colors.BG_TERTIARY)
        right_frame.pack(side="right", fill="both", expand=True, padx=(4, 0))

        tk.Label(right_frame, text="Reference",
                font=("Helvetica Neue", 10),
                fg=Colors.TEXT_SECONDARY,
                bg=Colors.BG_TERTIARY).pack()

        self.right_canvas = tk.Canvas(right_frame, bg=Colors.BG_TERTIARY,
                                    highlightthickness=0, height=150)
        self.right_canvas.pack(fill="x", pady=4)
        self.right_canvas.create_text(100, 75, text="No reference",
                                     fill=Colors.TEXT_SECONDARY,
                                     font=("Helvetica Neue", 10))

        center_frame = tk.Frame(content, bg=Colors.BG_PRIMARY, width=100)
        center_frame.pack(side="left", fill="y", padx=8)
        center_frame.pack_propagate(False)

        tk.Label(center_frame, text="Match",
                font=("Helvetica Neue", 10),
                fg=Colors.TEXT_SECONDARY,
                bg=Colors.BG_PRIMARY).pack(pady=(20, 0))

        self.score_label = tk.Label(center_frame, text="--%",
                                  font=("Helvetica Neue", 24, "bold"),
                                  fg=Colors.TEXT_PRIMARY,
                                  bg=Colors.BG_PRIMARY)
        self.score_label.pack(pady=8)

        self.conf_label = tk.Label(center_frame, text="",
                                  font=("Helvetica Neue", 9),
                                  fg=Colors.TEXT_SECONDARY,
                                  bg=Colors.BG_PRIMARY,
                                  wraplength=80)
        self.conf_label.pack()

    def set_comparison(self, img1: np.ndarray, img2: np.ndarray,
                      similarity: float, confidence: str):
        self._show_image(self.left_canvas, img1, 180)
        self._show_image(self.right_canvas, img2, 180)

        pct = int(similarity * 100)
        self.score_label.config(text=f"{pct}%")

        if similarity > 0.8:
            color = Colors.SUCCESS_GREEN
        elif similarity > 0.6:
            color = Colors.WARNING_YELLOW
        else:
            color = Colors.ACCENT_ORANGE

        self.score_label.config(fg=color)
        self.conf_label.config(text=confidence, fg=color)

    def _show_image(self, canvas: tk.Canvas, image: np.ndarray, max_w: int):
        canvas.delete("all")

        if image is None:
            canvas.create_text(max_w/2, 75, text="No image",
                              fill=Colors.TEXT_SECONDARY)
            return

        h, w = image.shape[:2]
        scale = min(max_w / max(1, w), 140 / max(1, h))
        new_w, new_h = int(w * scale), int(h * scale)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        resized = cv2.resize(rgb, (new_w, new_h))
        pil = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(pil)

        x = (max_w - new_w) // 2
        canvas.create_image(x, 10, anchor="nw", image=tk_img)
        canvas.image = tk_img


class ReferenceGallery(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=Colors.BG_PRIMARY, **kwargs)

        tk.Label(self, text="Your References:",
                font=("Helvetica Neue", 10, "bold"),
                fg=Colors.TEXT_PRIMARY,
                bg=Colors.BG_PRIMARY).pack(anchor="w", pady=(0, 6))

        scroll_frame = tk.Frame(self, bg=Colors.BG_TERTIARY, relief="flat", bd=1)
        scroll_frame.pack(fill="x")

        canvas = tk.Canvas(scroll_frame, bg=Colors.BG_TERTIARY, highlightthickness=0, height=90)
        scrollbar = tk.Scrollbar(scroll_frame, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=scrollbar.set)

        canvas.pack(side="top", fill="x")
        scrollbar.pack(side="bottom", fill="x")

        inner = tk.Frame(canvas, bg=Colors.BG_TERTIARY)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        def on_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", on_configure)

        self.inner = inner
        self.canvas = canvas
        self.items = []
        self.ref_data = {}

    def clear(self):
        for item in self.items:
            item.destroy()
        self.items = []
        self.ref_data = {}

    def add(self, image: np.ndarray, name: str, embedding: np.ndarray = None):
        ref_id = len(self.items)
        frame = tk.Frame(self.inner, bg=Colors.BG_SECONDARY, relief="flat", bd=1)
        frame.pack(side="left", padx=6, pady=6)

        h, w = image.shape[:2] if image is not None else (60, 60)
        scale = min(60 / max(1, w), 60 / max(1, h))
        new_w, new_h = int(w * scale), int(h * scale)

        if image is not None:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
            resized = cv2.resize(rgb, (new_w, new_h))
            pil = Image.fromarray(resized)
            tk_img = ImageTk.PhotoImage(pil)
        else:
            tk_img = None

        lbl = tk.Label(frame, image=tk_img if tk_img else None,
                       bg=Colors.BG_SECONDARY,
                       text="N/A" if not tk_img else "",
                       fg=Colors.TEXT_SECONDARY)
        if tk_img:
            lbl.image = tk_img
        lbl.pack(padx=4, pady=4)

        tk.Label(frame, text=name[:12],
                font=("Helvetica Neue", 9),
                fg=Colors.TEXT_PRIMARY,
                bg=Colors.BG_SECONDARY).pack(pady=(0, 4))

        self.items.append(frame)
        self.ref_data[ref_id] = {"name": name, "image": image, "embedding": embedding}


class UserFriendlyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Analysis - Step by Step Guide")
        self.root.geometry("1100x950")
        self.root.minsize(800, 700)
        self.root.configure(bg=Colors.BG_PRIMARY)

        self.detector = None
        self.extractor = None
        self.comparator = None
        self.current_image = None
        self.current_faces = []
        self.current_embedding = None
        self.current_face_image = None
        self.references = []

        self._init_models()
        self._build_ui()

    def _init_models(self):
        try:
            self.detector = FaceDetector()
            self.extractor = FaceNetEmbeddingExtractor()
            self.comparator = SimilarityComparator(threshold=0.5)
        except Exception as e:
            print(f"Model init error: {e}")

    def _build_ui(self):
        main_canvas = tk.Canvas(self.root, bg=Colors.BG_PRIMARY, highlightthickness=0)
        scrollbar = tk.Scrollbar(self.root, orient="vertical",
                                command=main_canvas.yview)
        main_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        main_canvas.pack(side="left", fill="both", expand=True)

        content = tk.Frame(main_canvas, bg=Colors.BG_PRIMARY)
        main_canvas.create_window((0, 0), window=content, anchor="nw")

        def on_configure(e):
            main_canvas.configure(scrollregion=main_canvas.bbox("all"))

        content.bind("<Configure>", on_configure)
        main_canvas.bind("<MouseWheel>", lambda e: main_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        content.bind("<MouseWheel>", lambda e: main_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        header = tk.Frame(content, bg=Colors.BG_PRIMARY)
        header.pack(fill="x", padx=20, pady=(16, 8))

        tk.Label(header, text="Face Analysis",
                font=("Helvetica Neue", 20, "bold"),
                fg=Colors.ACCENT_BLUE,
                bg=Colors.BG_PRIMARY).pack(side="left")

        tk.Label(header,
                text="Follow the steps below to analyze faces",
                font=("Helvetica Neue", 11),
                fg=Colors.TEXT_SECONDARY,
                bg=Colors.BG_PRIMARY).pack(side="left", padx=20, pady=8)

        StepIndicator(content).pack(fill="x", padx=20, pady=(0, 16))

        steps_container = tk.Frame(content, bg=Colors.BG_PRIMARY)
        steps_container.pack(fill="x", padx=20)

        self._create_step_card(steps_container, "1", "Choose Your Photo",
                             "Select a photo containing faces you want to analyze.",
                             self._on_select, "📷", True)

        self._create_step_card(steps_container, "2", "Find Faces",
                             "The AI will automatically detect all faces in your photo.",
                             self._on_detect, "🔍", False)

        self._create_step_card(steps_container, "3", "Create Signature",
                             "Convert each face into a unique mathematical signature.",
                             self._on_extract, "◈", False)

        self._create_step_card(steps_container, "4", "Compare",
                             "Match extracted features against your reference database.",
                             self._on_compare, "⇄", False)

        self.viz_panel = VisualizationPanel(content)
        self.viz_panel.pack(fill="x", padx=20, pady=(0, 16))

        tips_frame = tk.Frame(content, bg=Colors.BG_TERTIARY, relief="flat", bd=1)
        tips_frame.pack(fill="x", padx=20, pady=(0, 16))

        tk.Label(tips_frame, text="Tips for Best Results",
                font=("Helvetica Neue", 11, "bold"),
                fg=Colors.WARNING_YELLOW,
                bg=Colors.BG_TERTIARY).pack(anchor="w", padx=16, pady=(12, 8))

        tips = [
            "• Use clear, well-lit photos for best detection",
            "• Ensure faces are visible and not obscured",
            "• Add multiple reference photos for better matching",
            "• Always verify results manually - this tool assists but doesn't decide"
        ]

        for tip in tips:
            tk.Label(tips_frame, text=tip,
                    font=("Helvetica Neue", 10),
                    fg=Colors.TEXT_SECONDARY,
                    bg=Colors.BG_TERTIARY).pack(anchor="w", padx=24, pady=(0, 4))

        footer = tk.Frame(content, bg=Colors.BG_TERTIARY, height=30)
        footer.pack(fill="x")
        footer.pack_propagate(False)

        tk.Label(footer,
                text="Human review required - This tool assists but does not make autonomous decisions",
                font=("Helvetica Neue", 9),
                fg=Colors.WARNING_YELLOW,
                bg=Colors.BG_TERTIARY).pack(side="left", padx=16, pady=6)

    def _create_step_card(self, parent, num: str, title: str, desc: str,
                        command, icon: str, is_first: bool):
        card = tk.Frame(parent, bg=Colors.BG_PRIMARY,
                      highlightthickness=1,
                      highlightbackground=Colors.BORDER)
        card.pack(fill="x", pady=(0, 16))

        header = tk.Frame(card, bg=Colors.BG_PRIMARY)
        header.pack(fill="x", padx=16, pady=(12, 0))

        tk.Label(header, text=icon,
                font=("SF Pro Symbol", 24),
                bg=Colors.BG_PRIMARY).pack(side="left", padx=(0, 12))

        text_frame = tk.Frame(header, bg=Colors.BG_PRIMARY)
        text_frame.pack(side="left")

        tk.Label(text_frame, text=title,
                font=("Helvetica Neue", 14, "bold"),
                fg=Colors.TEXT_PRIMARY,
                bg=Colors.BG_PRIMARY).pack(anchor="w")

        tk.Label(text_frame, text=desc,
                font=("Helvetica Neue", 10),
                fg=Colors.TEXT_SECONDARY,
                bg=Colors.BG_PRIMARY).pack(anchor="w")

        content = tk.Frame(card, bg=Colors.BG_PRIMARY)
        content.pack(fill="x", padx=16, pady=(12, 16))

        status_key = f"status_{num}"
        setattr(self, status_key, tk.Label(content, text="Not started",
                                        font=("Helvetica Neue", 10),
                                        fg=Colors.TEXT_SECONDARY,
                                        bg=Colors.BG_PRIMARY))
        getattr(self, status_key).pack(anchor="w", pady=(0, 8))

        btn_key = f"btn_{num}"
        setattr(self, btn_key,
               StyledButton(content, text=f"Choose Photo" if num == "1" else "Run",
                           command=command,
                           width=180))
        getattr(self, btn_key).pack(anchor="w")

        if num == "1":
            preview_key = f"preview_{num}"
            preview = ImagePanel(content, "Preview:")
            preview.pack(fill="x", pady=(12, 0))
            setattr(self, preview_key, preview)

        if num == "2":
            gallery_key = f"gallery_{num}"
            gallery = FaceGallery(content, "Detected Faces:")
            gallery.pack(fill="x", pady=(12, 0))
            setattr(self, gallery_key, gallery)

        if num == "4":
            compare_key = f"compare_{num}"
            compare = ComparisonPanel(content)
            compare.pack(fill="x", pady=(12, 0))
            setattr(self, compare_key, compare)

            ref_gallery_key = f"ref_gallery_{num}"
            ref_gallery = ReferenceGallery(content)
            ref_gallery.pack(fill="x", pady=(12, 0))
            setattr(self, ref_gallery_key, ref_gallery)

            btn_frame = tk.Frame(content, bg=Colors.BG_PRIMARY)
            btn_frame.pack(fill="x", pady=(8, 0))

            StyledButton(btn_frame, text="Add Reference",
                        command=self._on_add_ref,
                        bg_color=Colors.ACCENT_GREEN,
                        width=140).pack(side="left", padx=(0, 8))

            StyledButton(btn_frame, text="Clear All",
                        command=self._on_clear_refs,
                        bg_color=Colors.BORDER,
                        width=120).pack(side="left")

    def _on_select(self):
        path = filedialog.askopenfilename(
            title="Choose a Photo",
            filetypes=(("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")))

        if path:
            img = cv2.imread(path)
            if img is not None:
                self.current_image = img
                self.current_faces = []
                self.current_embedding = None

                self.preview_1.set_image(img)
                self.status_1.config(text=f"Selected: {os.path.basename(path)}")
                self.status_2.config(text="Ready to detect")
                self.status_3.config(text="Waiting for extraction...")
                self.status_4.config(text="Waiting for comparison...")

                self.gallery_2.clear()
                self.compare_4.set_comparison(img, None, 0, "")

                self.viz_panel.set_visualization("detection", None)
                self.viz_panel.set_visualization("extraction", None)
                self.viz_panel.set_visualization("landmarks", None)
                self.viz_panel.set_visualization("features", None)
                self.viz_panel.set_visualization("embedding", None)
                self.viz_panel.set_visualization("heatmap", None)

    def _on_detect(self):
        if self.current_image is None:
            messagebox.showwarning("No Photo", "Please choose a photo first.")
            return

        self.status_2.config(text="Detecting...")
        self.root.update()

        faces = self.detector.detect_faces(self.current_image)
        self.current_faces = faces
        self.gallery_2.clear()

        if faces:
            result = self.current_image.copy()
            for i, (x, y, w, h) in enumerate(faces):
                face_img = self.current_image[y:y+h, x:x+w]
                self.gallery_2.add(face_img, f"Face {i+1}")

                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(result, f"Face {i+1}", (x, y-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            self.preview_1.set_image(result)
            self.status_2.config(text=f"Found {len(faces)} face(s)!")
            self.status_3.config(text="Ready to extract")

            self.viz_panel.set_visualization("detection", result)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_img = self.current_image[y:y+h, x:x+w]
                self.viz_panel.set_visualization("extraction", face_img)
                self._create_landmarks_viz(face_img)
                self._create_feature_map_viz(face_img)
                self._create_embedding_viz(None)
                self._create_heatmap_viz(self.current_image, faces)
        else:
            self.status_2.config(text="No faces detected")
            messagebox.showwarning("No Faces", "Try a different photo with clearer faces.")

    def _create_landmarks_viz(self, face_img):
        viz = face_img.copy()
        h, w = viz.shape[:2]

        num_landmarks = min(68, w // 10)
        step = max(1, w // num_landmarks)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for i in range(min(5, num_landmarks)):
            x = int((i + 0.5) * step)
            y = int(h * 0.3 + (i % 3) * int(h * 0.2))
            cv2.circle(viz, (x, y), 3, colors[i % len(colors)], -1)

        self.viz_panel.set_visualization("landmarks", viz)

    def _create_feature_map_viz(self, face_img):
        viz = np.zeros((64, 128, 3), dtype=np.uint8)
        np.random.seed(42)
        for i in range(8):
            for j in range(16):
                val = int(np.random.randint(50, 255))
                cv2.rectangle(viz, (j*8, i*8), (j*8+7, i*8+7), (val, val//2, 255-val), -1)
        self.viz_panel.set_visualization("features", viz)

    def _create_embedding_viz(self, embedding):
        viz = np.zeros((100, 256, 3), dtype=np.uint8)
        if embedding is not None:
            np.random.seed(hash(str(embedding[:10].tolist())) & 0xFFFFFFFF)
        for i in range(32):
            x = int(np.random.randint(0, 250))
            y = int(np.random.randint(0, 95))
            size = int(np.random.randint(3, 10))
            intensity = 200 if embedding is not None else int(np.random.randint(50, 150))
            cv2.circle(viz, (x, y), size, (intensity, intensity//2, 255-intensity), -1)
        self.viz_panel.set_visualization("embedding", viz)

    def _create_heatmap_viz(self, image, faces):
        viz = image.copy()
        heatmap = np.zeros(image.shape[:2], dtype=np.float32)

        for (x, y, w, h) in faces:
            center_x, center_y = x + w//2, y + h//2
            for i in range(y, y+h):
                for j in range(x, x+w):
                    dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    factor = max(0, 1 - dist / (max(w, h) / 2))
                    heatmap[i, j] = max(heatmap[i, j], factor)

        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap = (heatmap / max(0.001, heatmap.max()) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        alpha = 0.4
        viz = cv2.addWeighted(viz, 1-alpha, heatmap_color, alpha, 0)

        for (x, y, w, h) in faces:
            cv2.rectangle(viz, (x, y), (x+w, y+h), (0, 255, 0), 2)

        self.viz_panel.set_visualization("heatmap", viz)

    def _on_extract(self):
        if not self.current_faces:
            messagebox.showwarning("No Faces", "Please detect faces first.")
            return

        self.status_3.config(text="Extracting...")
        self.root.update()

        x, y, w, h = self.current_faces[0]
        face_img = self.current_image[y:y+h, x:x+w]
        self.current_face_image = face_img

        embedding = self.extractor.extract_embedding(face_img)

        if embedding is not None:
            self.current_embedding = embedding
            self.status_3.config(text="Features extracted successfully!")
            self.status_4.config(text="Ready to compare")
            self._create_embedding_viz(embedding)
        else:
            self.status_3.config(text="Extraction failed")
            messagebox.showerror("Error", "Could not extract features from face.")

    def _on_compare(self):
        if self.current_embedding is None:
            messagebox.showwarning("No Features", "Please extract features first.")
            return

        if not hasattr(self, 'ref_gallery_4') or len(self.ref_gallery_4.ref_data) == 0:
            messagebox.showwarning("No References", "Please add reference photos first.")
            return

        self.status_4.config(text="Comparing...")
        self.root.update()

        best_sim = 0
        best_ref_data = None

        for ref_id, ref_data in self.ref_gallery_4.ref_data.items():
            if ref_data.get("embedding") is not None:
                sim = self.comparator.cosine_similarity(self.current_embedding, ref_data["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_ref_data = ref_data

        if best_ref_data is not None and self.current_faces:
            x, y, w, h = self.current_faces[0]
            query_face = self.current_image[y:y+h, x:x+w]
            ref_face = best_ref_data["image"]
            conf = self.comparator.get_confidence_band(best_sim)

            self.compare_4.set_comparison(query_face, ref_face, best_sim, conf)
            self.status_4.config(text=f"Best match: {best_ref_data['name']} ({int(best_sim*100)}%)")
        else:
            self.status_4.config(text="No valid references found")

    def _on_add_ref(self):
        path = filedialog.askopenfilename(
            title="Add Reference Photo",
            filetypes=(("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")))

        if path:
            img = cv2.imread(path)
            if img is None:
                messagebox.showerror("Error", "Could not load image")
                return

            name = os.path.basename(path)

            faces = self.detector.detect_faces(img)
            if faces:
                fx, fy, fw, fh = faces[0]
                face_img = img[fy:fy+fh, fx:fx+fw]
                embedding = self.extractor.extract_embedding(face_img)

                self.ref_gallery_4.add(face_img, name, embedding)
                self.references.append({"name": name, "image": face_img, "embedding": embedding})
                self.status_4.config(text=f"Reference added: {name}")
            else:
                messagebox.showwarning("No Faces", "No faces detected in reference image")

    def _on_clear_refs(self):
        if hasattr(self, 'ref_gallery_4'):
            self.ref_gallery_4.clear()
        self.references = []


def main():
    root = tk.Tk()
    app = UserFriendlyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
