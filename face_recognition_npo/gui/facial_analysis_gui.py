#!/usr/bin/env python3
"""
NGO Facial Image Analysis System - Apple-Styled GUI
A clean, modern interface for ethical facial image analysis.
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
from src.reference import ReferenceImageManager


class AppleStyle:
    """Apple-inspired color palette and styling constants."""
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
    BORDER_COLOR = "#D2D2D7"
    SUCCESS_GREEN = "#30D158"
    WARNING_YELLOW = "#FFD60A"
    ERROR_RED = "#FF453A"
    PROCESSING_BLUE = "#5856D6"

    @classmethod
    def get_font(cls, size: int = 11, weight: str = "normal") -> tuple:
        if sys.platform == "darwin":
            return ("SF Pro", size, weight)
        elif sys.platform == "win32":
            return ("Segoe UI", size, weight)
        else:
            return ("Helvetica Neue", size, weight)


class Tooltip:
    """Tooltip manager for widgets."""
    def __init__(self, widget, text=""):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
        self.widget.bind("<Motion>", self.update_position)

    def update_position(self, event=None):
        if self.tooltip_window:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
            self.tooltip_window.wm_geometry(f"+{x}+{y}")

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        self.tooltip_window.configure(bg=AppleStyle.TEXT_PRIMARY)
        label = tk.Label(self.tooltip_window, text=self.text,
                        font=AppleStyle.get_font(10),
                        fg=AppleStyle.BG_PRIMARY,
                        bg=AppleStyle.TEXT_PRIMARY,
                        padx=10, pady=6, wraplength=280, justify="left")
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class LoadingIndicator(tk.Frame):
    """Animated loading indicator."""
    def __init__(self, parent, text="Processing...", **kwargs):
        super().__init__(parent, bg=AppleStyle.BG_PRIMARY, **kwargs)
        self.text = text
        self.dots = 0
        self.label = tk.Label(self, text=text + "...",
                            font=AppleStyle.get_font(11),
                            fg=AppleStyle.PROCESSING_BLUE,
                            bg=AppleStyle.BG_PRIMARY)
        self.label.pack(side="left", padx=(0, 8))
        self.animation_id = None
    
    def start(self, duration_ms=None):
        self.dots = 0
        self._animate()
        if duration_ms:
            self.root.after(duration_ms, self.stop)
    
    def _animate(self):
        dots = "." * ((self.dots % 4))
        self.label.config(text=self.text + dots)
        self.dots += 1
        self.animation_id = self.root.after(300, self._animate)
    
    def stop(self):
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None
        self.label.config(text=self.text + " ✓")
    
    def set_text(self, text: str):
        self.text = text
        self.label.config(text=text)
    
    @property
    def root(self):
        return self.winfo_toplevel()


class StatusCard(tk.Frame):
    """Status card with icon and message."""
    def __init__(self, parent, icon="✓", message="", status="success", **kwargs):
        super().__init__(parent, bg=AppleStyle.BG_PRIMARY, **kwargs)
        colors = {
            "success": AppleStyle.SUCCESS_GREEN,
            "warning": AppleStyle.WARNING_YELLOW,
            "error": AppleStyle.ERROR_RED,
            "info": AppleStyle.PROCESSING_BLUE,
            "processing": AppleStyle.PROCESSING_BLUE
        }
        color = colors.get(status, AppleStyle.SUCCESS_GREEN)
        self.icon_label = tk.Label(self, text=icon, font=AppleStyle.get_font(14),
                                  fg=color, bg=AppleStyle.BG_PRIMARY)
        self.icon_label.pack(side="left", padx=(0, 8))
        self.msg_label = tk.Label(self, text=message, font=AppleStyle.get_font(10),
                                 fg=AppleStyle.TEXT_PRIMARY, bg=AppleStyle.BG_PRIMARY)
        self.msg_label.pack(side="left")
        self.status = status
        self.icon = icon
    
    def update(self, icon=None, message=None, status=None):
        if icon:
            self.icon = icon
        if message:
            self.msg_label.config(text=message)
        if status:
            self.status = status
        colors = {
            "success": AppleStyle.SUCCESS_GREEN,
            "warning": AppleStyle.WARNING_YELLOW,
            "error": AppleStyle.ERROR_RED,
            "info": AppleStyle.PROCESSING_BLUE,
            "processing": AppleStyle.PROCESSING_BLUE
        }
        color = colors.get(self.status, AppleStyle.SUCCESS_GREEN)
        self.icon_label.config(text=self.icon, fg=color)


class StyledButton(tk.Canvas):
    """Apple-styled rounded button widget."""
    def __init__(self, parent, text="", command=None, width=120, height=36,
                 bg_color="#007AFF", fg_color="#FFFFFF", hover_color="#0066CC",
                 font=None, corner_radius=8, **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg=AppleStyle.BG_PRIMARY, highlightthickness=0, **kwargs)
        self.command = command
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.hover_color = hover_color
        self.corner_radius = corner_radius
        self.font = font or AppleStyle.get_font(12, "normal")
        self._text = text
        self._enabled = True
        self.tooltip_text = ""
        self._draw_button(bg_color)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _draw_button(self, color):
        self.delete("all")
        w, h, r = self.winfo_reqwidth(), self.winfo_reqheight(), self.corner_radius
        self.create_arc(0, 0, r*2, r*2, start=90, extent=90, fill=color, outline=color)
        self.create_arc(w-r*2, 0, w, r*2, start=0, extent=90, fill=color, outline=color)
        self.create_arc(0, h-r*2, r*2, h, start=180, extent=90, fill=color, outline=color)
        self.create_arc(w-r*2, h-r*2, w, h, start=270, extent=90, fill=color, outline=color)
        self.create_rectangle(r, 0, w-r, h, fill=color, outline=color)
        self.create_rectangle(0, r, w, h-r, fill=color, outline=color)
        self.create_text(w/2, h/2, text=self._text, fill=self.fg_color, font=self.font, tags="text")

    def set_tooltip(self, text: str):
        self.tooltip_text = text
        if text:
            Tooltip(self, text)

    def _on_enter(self, event):
        if self._enabled:
            self._draw_button(self.hover_color)

    def _on_leave(self, event):
        if self._enabled:
            self._draw_button(self.bg_color)

    def _on_click(self, event):
        if self._enabled:
            self._draw_button(self.hover_color)

    def _on_release(self, event):
        if self._enabled:
            self._draw_button(self.bg_color)
            if self.command:
                self.command()


class ImageViewer(tk.Frame):
    """Professional image viewer."""
    def __init__(self, parent, title="", max_height=280, **kwargs):
        super().__init__(parent, bg=AppleStyle.BG_TERTIARY, **kwargs)
        self.max_height = max_height
        header = tk.Frame(self, bg=AppleStyle.BG_TERTIARY)
        header.pack(fill="x", padx=8, pady=(6, 4))
        self.title_label = tk.Label(header, text=title, font=AppleStyle.get_font(11, "bold"),
                                   fg=AppleStyle.TEXT_PRIMARY, bg=AppleStyle.BG_TERTIARY)
        self.title_label.pack(side="left")
        self.canvas = tk.Canvas(self, bg=AppleStyle.BG_TERTIARY, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._show_placeholder()
        self.current_image = None
        self.current_tk = None
    
    def _show_placeholder(self):
        self.placeholder = self.canvas.create_text(
            self.winfo_reqwidth()/2, self.winfo_reqheight()/2,
            text="◉\n\nNo image loaded", font=AppleStyle.get_font(12),
            fill=AppleStyle.TEXT_SECONDARY, tags="placeholder")
    
    def set_title(self, title: str):
        self.title_label.config(text=title)
    
    def load_image(self, image: np.ndarray) -> bool:
        if image is None:
            return False
        self.current_image = image
        self._render_image(image)
        return True
    
    def _render_image(self, image: np.ndarray):
        self.canvas.delete("all")
        if image is None:
            self._show_placeholder()
            return
        h, w = image.shape[:2]
        canvas_w = max(self.winfo_reqwidth() - 16, 100)
        canvas_h = self.max_height
        scale = min(canvas_w / max(1, w), canvas_h / max(1, h), 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        image_resized = cv2.resize(image_rgb, (new_w, new_h))
        image_pil = Image.fromarray(image_resized)
        self.current_tk = ImageTk.PhotoImage(image_pil)
        x = (canvas_w - new_w) // 2 + 8
        y = (canvas_h - new_h) // 2 + 8
        self.canvas.create_image(x, y, anchor="nw", image=self.current_tk)


class ImageGallery(tk.Frame):
    """Gallery for displaying multiple images with thumbnails."""
    def __init__(self, parent, title="", **kwargs):
        super().__init__(parent, bg=AppleStyle.BG_PRIMARY, **kwargs)
        header = tk.Frame(self, bg=AppleStyle.BG_PRIMARY)
        header.pack(fill="x", pady=(0, 8))
        tk.Label(header, text=title, font=AppleStyle.get_font(11, "bold"),
                fg=AppleStyle.TEXT_PRIMARY, bg=AppleStyle.BG_PRIMARY).pack(side="left")
        self.scroll_frame = tk.Frame(self, bg=AppleStyle.BG_TERTIARY, relief="flat", bd=1)
        self.scroll_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.scroll_frame, bg=AppleStyle.BG_TERTIARY, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.scroll_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.scrollbar.pack(side="bottom", fill="x")
        self.inner_frame = tk.Frame(self.canvas, bg=AppleStyle.BG_TERTIARY)
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.inner_frame.bind("<Configure>", self._on_frame_configure)
        self.items = []
    
    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def clear(self):
        for item in self.items:
            item.destroy()
        self.items = []
    
    def add_image(self, image: np.ndarray, label: str = "", tooltip: str = "", match_info: str = ""):
        frame = tk.Frame(self.inner_frame, bg=AppleStyle.BG_SECONDARY, relief="flat", bd=1)
        frame.pack(side="left", padx=6, pady=6)
        h, w = image.shape[:2] if image is not None else (100, 100)
        scale = min(100 / max(1, w), 100 / max(1, h), 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
            image_resized = cv2.resize(image_rgb, (new_w, new_h))
            image_pil = Image.fromarray(image_resized)
            image_tk = ImageTk.PhotoImage(image_pil)
        else:
            image_tk = None
        img_label = tk.Label(frame, image=image_tk if image_tk else None, 
                           bg=AppleStyle.BG_SECONDARY, cursor="hand2")
        if image_tk:
            img_label.image = image_tk
        else:
            img_label.config(text="N/A", fg=AppleStyle.TEXT_SECONDARY)
        img_label.pack(padx=4, pady=(4, 2))
        tk.Label(frame, text=label, font=AppleStyle.get_font(9),
                fg=AppleStyle.TEXT_SECONDARY, bg=AppleStyle.BG_SECONDARY).pack(pady=(0, 2))
        if match_info:
            tk.Label(frame, text=match_info, font=AppleStyle.get_font(9, "bold"),
                    fg=AppleStyle.SUCCESS_GREEN, bg=AppleStyle.BG_SECONDARY).pack(pady=(0, 4))
        if tooltip:
            Tooltip(img_label, tooltip)
        self.items.append(frame)


class ResultCard(tk.Frame):
    """Card for displaying comparison results."""
    def __init__(self, parent, ref_id: str, similarity: float, confidence: str, **kwargs):
        super().__init__(parent, bg=AppleStyle.BG_SECONDARY, relief="flat", bd=1, **kwargs)
        colors = {
            "High confidence": AppleStyle.SUCCESS_GREEN,
            "Moderate confidence": AppleStyle.WARNING_YELLOW,
            "Low confidence": AppleStyle.ACCENT_ORANGE,
            "Insufficient confidence": AppleStyle.TEXT_SECONDARY
        }
        bar_color = colors.get(confidence, AppleStyle.TEXT_SECONDARY)
        tk.Label(self, text=ref_id, font=AppleStyle.get_font(11, "bold"),
                fg=AppleStyle.TEXT_PRIMARY, bg=AppleStyle.BG_SECONDARY).pack(anchor="w", padx=10, pady=(10, 4))
        bar_frame = tk.Frame(self, bg=AppleStyle.BG_TERTIARY, height=8)
        bar_frame.pack(fill="x", padx=10, pady=(0, 4))
        bar_frame.pack_propagate(False)
        bar_width = int(similarity * 100)
        bar = tk.Frame(bar_frame, bg=bar_color, width=bar_width, height=6)
        bar.place(x=2, y=1)
        tk.Label(self, text=f"{similarity:.1%} — {confidence}", font=AppleStyle.get_font(9),
                fg=bar_color, bg=AppleStyle.BG_SECONDARY).pack(anchor="w", padx=10, pady=(0, 10))


class FacialAnalysisGUI:
    """Main GUI application with Apple-inspired design."""

    def __init__(self, root):
        self.root = root
        self.root.title("NGO Facial Image Analysis")
        self.root.geometry("1300x900")
        self.root.minsize(1200, 800)
        self.root.configure(bg=AppleStyle.BG_PRIMARY)
        
        self.detector = None
        self.extractor = None
        self.comparator = None
        self.reference_manager = None
        
        self.current_image = None
        self.current_faces = []
        self.current_embedding = None
        self.reference_data = {}
        
        self.reference_embeddings = []
        self.reference_ids = []
        
        self.webcam_active = False
        self.webcam_cap = None
        
        self.loading_indicator = None
        
        self._build_ui()
        self._init_models()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_models(self):
        """Initialize all required models."""
        try:
            self.detector = FaceDetector()
            self.extractor = FaceNetEmbeddingExtractor()
            self.comparator = SimilarityComparator(threshold=0.5)
            self.reference_manager = ReferenceImageManager(embedding_extractor=self.extractor)
            self._load_references()
            self.log_message("Models initialized successfully", "success")
        except Exception as e:
            self.log_message(f"Model initialization: {e}", "warning")

    def _load_references(self):
        """Load existing reference embeddings."""
        try:
            ref_embeddings, ref_ids = self.reference_manager.get_reference_embeddings()
            for ref_id, embedding in zip(ref_ids, ref_embeddings):
                if ref_id not in self.reference_ids:
                    self.reference_ids.append(ref_id)
                    self.reference_embeddings.append(embedding)
            self._refresh_references()
            self.log_message(f"Loaded {len(self.reference_ids)} reference images")
        except Exception as e:
            self.log_message(f"Could not load references: {e}")

    def _build_ui(self):
        """Build the complete UI layout."""
        self._create_header()
        self._create_notebook()
        self._create_footer()

    def _create_header(self):
        """Create the application header."""
        header = tk.Frame(self.root, bg=AppleStyle.BG_PRIMARY, height=52)
        header.pack(side="top", fill="x", padx=20)
        header.pack_propagate(False)
        tk.Label(header, text="◈", font=("SF Pro Symbol", 20),
                fg=AppleStyle.ACCENT_BLUE, bg=AppleStyle.BG_PRIMARY).pack(side="left", padx=(0, 8))
        tk.Label(header, text="Facial Image Analysis",
                font=AppleStyle.get_font(16, "bold"),
                fg=AppleStyle.TEXT_PRIMARY, bg=AppleStyle.BG_PRIMARY).pack(side="left")
        tk.Label(header, text="Ethical NGO Documentation Verification",
                font=AppleStyle.get_font(10),
                fg=AppleStyle.TEXT_SECONDARY, bg=AppleStyle.BG_PRIMARY).pack(side="right", pady=14)

    def _create_notebook(self):
        """Create tabbed interface."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        self._create_analyze_tab()
        self._create_visualizations_tab()
        self._create_references_tab()
        self._create_functions_tab()
        self._create_logs_tab()

    def _create_analyze_tab(self):
        """Create the main analysis tab."""
        tab = tk.Frame(self.notebook, bg=AppleStyle.BG_PRIMARY, padx=16, pady=16)
        self.notebook.add(tab, text="  Analyze  ")
        
        left_panel = tk.Frame(tab, bg=AppleStyle.BG_PRIMARY)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 12))
        
        right_panel = tk.Frame(tab, bg=AppleStyle.BG_SECONDARY, width=280)
        right_panel.pack(side="right", fill="y", padx=(12, 0))
        right_panel.pack_propagate(False)
        
        self._create_workflow_buttons(left_panel)
        self._create_image_displays(left_panel)
        self._create_results_section(left_panel)
        self._create_references_panel(right_panel)

    def _create_visualizations_tab(self):
        """Create visualizations gallery tab."""
        tab = tk.Frame(self.notebook, bg=AppleStyle.BG_PRIMARY, padx=16, pady=16)
        self.notebook.add(tab, text="  Visualizations  ")
        
        top_frame = tk.Frame(tab, bg=AppleStyle.BG_PRIMARY)
        top_frame.pack(fill="x", pady=(0, 12))
        tk.Label(top_frame, text="Analysis Visualizations",
                font=AppleStyle.get_font(18, "bold"),
                fg=AppleStyle.TEXT_PRIMARY, bg=AppleStyle.BG_PRIMARY).pack(side="left")
        
        btn_frame = tk.Frame(top_frame, bg=AppleStyle.BG_PRIMARY)
        btn_frame.pack(side="right")
        
        StyledButton(btn_frame, text="Load Dashboard",
                    command=self._load_dashboard,
                    bg_color=AppleStyle.ACCENT_PURPLE, width=130).pack(side="left", padx=(0, 8))
        
        StyledButton(btn_frame, text="Open All",
                    command=self._open_all_visualizations,
                    bg_color=AppleStyle.ACCENT_BLUE, width=100).pack(side="left")
        
        self.viz_gallery = ImageGallery(tab, "Available Visualizations")
        self.viz_gallery.pack(fill="both", expand=True)
        self._load_visualization_gallery()

    def _load_visualization_gallery(self):
        """Load visualization images into gallery."""
        viz_dir = "test_images"
        if not os.path.isdir(viz_dir):
            return
        
        viz_files = [
            ("kanye_west_3D_MESH.jpeg", "3D Mesh", "Facial landmark mesh overlay showing keypoints"),
            ("kanye_west_ACTIVATIONS.jpeg", "Activations", "Neural network activation patterns"),
            ("kanye_west_ADVERSARIAL.jpeg", "Adversarial", "Adversarial perturbation analysis"),
            ("kanye_west_ALIGNMENT.jpeg", "Alignment", "Face alignment visualization"),
            ("kanye_west_CONFIDENCE.jpeg", "Confidence", "Confidence heatmap"),
            ("kanye_west_FEATURE_IMPORTANCE.jpeg", "Features", "Feature importance visualization"),
            ("kanye_west_MULTISCALE.jpeg", "Multi-Scale", "Multi-scale feature detection"),
            ("kanye_west_SIMILARITY.jpeg", "Similarity", "Similarity comparison map"),
            ("kanye_west_biometric_visualization.jpeg", "Biometric", "Biometric capture visualization"),
        ]
        
        loaded = set()
        for filename, label, tooltip in viz_files:
            filepath = os.path.join(viz_dir, filename)
            if os.path.exists(filepath) and filename not in loaded:
                img = cv2.imread(filepath)
                if img is not None:
                    self.viz_gallery.add_image(img, label, tooltip)
                    loaded.add(filename)

    def _load_dashboard(self):
        """Load dashboard visualization."""
        dashboard_files = [
            "kanye_west_COMPLETE_DASHBOARD.jpeg",
            "kanye_west_dashboard.jpeg",
            "kanye_west_breakdown.jpeg",
        ]
        
        for filename in dashboard_files:
            filepath = os.path.join("test_images", filename)
            if os.path.exists(filepath):
                img = cv2.imread(filepath)
                if img is not None:
                    self.source_viewer.load_image(img)
                    self.source_viewer.set_title(f"Dashboard: {filename}")
                    self.log_message(f"Loaded dashboard: {filename}", "success")
                    return
        
        self.log_message("No dashboard file found", "warning")

    def _open_all_visualizations(self):
        """Open all visualization images."""
        viz_files = [
            "kanye_west_3D_MESH.jpeg", "kanye_west_ACTIVATIONS.jpeg", 
            "kanye_west_ADVERSARIAL.jpeg", "kanye_west_ALIGNMENT.jpeg",
            "kanye_west_CONFIDENCE.jpeg", "kanye_west_FEATURE_IMPORTANCE.jpeg",
            "kanye_west_MULTISCALE.jpeg", "kanye_west_SIMILARITY.jpeg",
            "kanye_west_biometric_visualization.jpeg",
        ]
        
        opened = 0
        for filename in viz_files:
            filepath = os.path.join("test_images", filename)
            if os.path.exists(filepath):
                img = cv2.imread(filepath)
                if img is not None:
                    try:
                        cv2.imshow(f"Visualization: {filename}", img)
                        opened += 1
                    except:
                        pass
        
        if opened > 0:
            self.log_message(f"Opened {opened} visualization windows", "success")
            messagebox.showinfo("Visualizations", f"Opened {opened} visualization windows.\nCheck your screen.")
        else:
            self.log_message("No visualizations found", "warning")

    def _create_functions_tab(self):
        """Create functions/terminal access tab."""
        tab = tk.Frame(self.notebook, bg=AppleStyle.BG_PRIMARY, padx=20, pady=20)
        self.notebook.add(tab, text="  Functions  ")
        
        tk.Label(tab, text="Advanced Functions & Terminal Access",
                font=AppleStyle.get_font(18, "bold"),
                fg=AppleStyle.TEXT_PRIMARY, bg=AppleStyle.BG_PRIMARY).pack(anchor="w", pady=(0, 16))
        
        # Functions grid
        funcs_frame = tk.Frame(tab, bg=AppleStyle.BG_PRIMARY)
        funcs_frame.pack(fill="both", expand=True)
        
        functions = [
            ("◈ Face Detection", "Detect faces in images", self._func_face_detection),
            ("◈ Extract Embeddings", "Generate 128-dim face embeddings", self._func_extract_embeddings),
            ("◈ Compare Faces", "Compare embeddings with references", self._func_compare),
            ("◈ Batch Processing", "Process multiple images at once", self._func_batch),
            ("◈ Generate Dashboard", "Create analysis dashboard", self._func_dashboard),
            ("◈ Export Report", "Export analysis results", self._func_export),
            ("◈ Clear References", "Remove all reference images", self._func_clear_refs),
            ("◈ View Test Images", "Browse test image gallery", self._func_test_images),
        ]
        
        for i, (name, desc, cmd) in enumerate(functions):
            row, col = divmod(i, 4)
            card = tk.Frame(funcs_frame, bg=AppleStyle.BG_SECONDARY, relief="flat", bd=1)
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            funcs_frame.grid_columnconfigure(col, weight=1)
            
            tk.Label(card, text=name, font=AppleStyle.get_font(12, "bold"),
                    fg=AppleStyle.ACCENT_BLUE, bg=AppleStyle.BG_SECONDARY).pack(pady=(12, 4))
            tk.Label(card, text=desc, font=AppleStyle.get_font(9),
                    fg=AppleStyle.TEXT_SECONDARY, bg=AppleStyle.BG_SECONDARY,
                    wraplength=180, justify="center").pack(pady=(0, 8), padx=8)
            
            StyledButton(card, text="Run", command=cmd,
                        bg_color=AppleStyle.ACCENT_GREEN, width=80, height=30).pack(pady=(0, 12))

    def _func_face_detection(self):
        """Function: Face Detection."""
        filepath = filedialog.askopenfilename(title="Select Image for Detection",
                filetypes=(("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")))
        if filepath:
            img = cv2.imread(filepath)
            if img is not None:
                faces = self.detector.detect_faces(img)
                result = img.copy()
                for i, (x, y, w, h) in enumerate(faces):
                    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(result, f"Face {i+1}", (x, y-8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Face Detection Result", result)
                self.log_message(f"Face detection: {len(faces)} face(s) in {os.path.basename(filepath)}", "success")
                messagebox.showinfo("Face Detection", f"Detected {len(faces)} face(s).\nResult shown in new window.")

    def _func_extract_embeddings(self):
        """Function: Extract Embeddings."""
        if not self.current_faces:
            messagebox.showwarning("No Faces", "Detect faces first.")
            return
        face = self.current_faces[0]
        x, y, w, h = face
        face_img = self.current_image[y:y+h, x:x+w]
        embedding = self.extractor.extract_embedding(face_img)
        if embedding is not None:
            self.log_message(f"Embedding extracted: 128-dim, norm={np.linalg.norm(embedding):.4f}", "success")
            messagebox.showinfo("Embedding", f"128-dimensional embedding extracted.\nNorm: {np.linalg.norm(embedding):.4f}")
        else:
            self.log_message("Embedding extraction failed", "error")

    def _func_compare(self):
        """Function: Compare Faces."""
        if not self.reference_ids:
            messagebox.showwarning("No References", "Add references first.")
            return
        if not self.current_embedding:
            messagebox.showwarning("No Embedding", "Extract embedding first.")
            return
        results = self.comparator.compare_embeddings(self.current_embedding, self.reference_embeddings, self.reference_ids)
        msg = "\n".join([f"{r[0]}: {r[1]:.1%} ({self.comparator.get_confidence_band(r[1])})" for r in results])
        self.log_message(f"Comparison: {len(results)} matches", "success")
        messagebox.showinfo("Comparison Results", msg if msg else "No matches above threshold.")

    def _func_batch(self):
        """Function: Batch Processing."""
        filepaths = filedialog.askopenfilenames(title="Select Images for Batch Processing",
                filetypes=(("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")))
        if filepaths:
            results = []
            for filepath in filepaths:
                img = cv2.imread(filepath)
                if img is not None:
                    faces = self.detector.detect_faces(img)
                    results.append((os.path.basename(filepath), len(faces)))
            msg = "\n".join([f"{name}: {count} face(s)" for name, count in results])
            self.log_message(f"Batch processed {len(results)} images", "success")
            messagebox.showinfo("Batch Results", msg)

    def _func_dashboard(self):
        """Function: Generate Dashboard."""
        self._load_dashboard()
        self.notebook.select(self._create_visualizations_tab_index if hasattr(self, '_create_visualizations_tab_index') else 1)

    def _func_export(self):
        """Function: Export Report."""
        self._on_export_report()

    def _func_clear_refs(self):
        """Function: Clear References."""
        if messagebox.askyesno("Confirm", f"Remove all {len(self.reference_ids)} references?"):
            self.reference_ids.clear()
            self.reference_embeddings.clear()
            self._refresh_references()
            self._refresh_ref_gallery()
            self.log_message("All references cleared", "info")

    def _func_test_images(self):
        """Function: View Test Images."""
        test_dir = "test_images"
        if os.path.isdir(test_dir):
            images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            msg = f"Available test images ({len(images)}):\n\n" + "\n".join(images[:15])
            if len(images) > 15:
                msg += f"\n... and {len(images) - 15} more"
            messagebox.showinfo("Test Images", msg)

    def _create_workflow_buttons(self, parent):
        """Create workflow action buttons."""
        btn_frame = tk.Frame(parent, bg=AppleStyle.BG_PRIMARY)
        btn_frame.pack(fill="x", pady=(0, 12))
        
        btn_select = StyledButton(btn_frame, text="1. Select Image",
                                command=self._on_select_image,
                                bg_color=AppleStyle.ACCENT_BLUE, width=130)
        btn_select.pack(side="left", padx=(0, 8))
        btn_select.set_tooltip("Open an image file for analysis")
        
        btn_detect = StyledButton(btn_frame, text="2. Detect Faces",
                                command=self._on_detect_faces,
                                bg_color=AppleStyle.ACCENT_PURPLE, width=130)
        btn_detect.pack(side="left", padx=(0, 8))
        btn_detect.set_tooltip("Find and locate all faces using AI detection")
        
        btn_extract = StyledButton(btn_frame, text="3. Extract Features",
                                command=self._on_extract,
                                bg_color=AppleStyle.ACCENT_ORANGE, width=120)
        btn_extract.pack(side="left", padx=(0, 8))
        btn_extract.set_tooltip("Convert detected face(s) into numerical embeddings")
        
        btn_compare = StyledButton(btn_frame, text="4. Compare",
                                command=self._on_compare,
                                bg_color=AppleStyle.ACCENT_GREEN, width=120)
        btn_compare.pack(side="left", padx=(0, 8))
        btn_compare.set_tooltip("Compare features against your reference images")

    def _create_image_displays(self, parent):
        """Create image display panels."""
        images_frame = tk.Frame(parent, bg=AppleStyle.BG_PRIMARY)
        images_frame.pack(fill="both", expand=True, pady=(0, 12))
        
        self.source_viewer = ImageViewer(images_frame, "Source Image", max_height=300)
        self.source_viewer.pack(fill="both", expand=True, pady=(0, 8))
        
        detected_frame = tk.Frame(images_frame, bg=AppleStyle.BG_PRIMARY)
        detected_frame.pack(fill="both", expand=True)
        
        self.faces_gallery = ImageGallery(detected_frame, "Detected Faces")
        self.faces_gallery.pack(fill="both", expand=True)

    def _create_results_section(self, parent):
        """Create results display section."""
        results_frame = tk.LabelFrame(parent, text=" Analysis Results ",
                                     font=AppleStyle.get_font(11, "bold"),
                                     fg=AppleStyle.TEXT_PRIMARY, bg=AppleStyle.BG_PRIMARY, bd=0)
        results_frame.pack(fill="x")
        
        self.results_container = tk.Frame(results_frame, bg=AppleStyle.BG_PRIMARY)
        self.results_container.pack(fill="both", expand=True, padx=8, pady=8)
        
        self.status_card = StatusCard(self.results_container, icon="◉",
                                    message="Select an image to begin analysis", status="info")
        self.status_card.pack(anchor="w")
        
        self.loading_indicator = LoadingIndicator(self.results_container, text="")
        self.loading_indicator.pack(anchor="w", pady=(8, 0))
        self.loading_indicator.stop()

    def _create_references_panel(self, parent):
        """Create references management panel."""
        tk.Label(parent, text="References", font=AppleStyle.get_font(13, "bold"),
                fg=AppleStyle.TEXT_PRIMARY, bg=AppleStyle.BG_SECONDARY).pack(anchor="w", padx=12, pady=(12, 4))
        
        self.refs_count_label = tk.Label(parent, text=f"{len(self.reference_ids)} loaded",
                                       font=AppleStyle.get_font(9), fg=AppleStyle.TEXT_SECONDARY,
                                       bg=AppleStyle.BG_SECONDARY)
        self.refs_count_label.pack(anchor="w", padx=12, pady=(0, 12))
        
        listbox_frame = tk.Frame(parent, bg=AppleStyle.BG_SECONDARY)
        listbox_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        
        scrollbar = tk.Scrollbar(listbox_frame, orient="vertical")
        self.refs_listbox = tk.Listbox(listbox_frame, font=AppleStyle.get_font(10),
                                       bg=AppleStyle.BG_TERTIARY, fg=AppleStyle.TEXT_PRIMARY,
                                       relief="flat", bd=0, selectbackground=AppleStyle.ACCENT_BLUE,
                                       yscrollcommand=scrollbar.set)
        self.refs_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.refs_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        
        btn_frame = tk.Frame(parent, bg=AppleStyle.BG_SECONDARY)
        btn_frame.pack(fill="x", padx=12, pady=(0, 12))
        
        btn_add = StyledButton(btn_frame, text="Add Reference", command=self._on_add_reference,
                             bg_color=AppleStyle.ACCENT_BLUE, width=110, height=32)
        btn_add.pack(fill="x", pady=(0, 6))
        btn_add.set_tooltip("Add a reference image for comparisons")
        
        btn_import = StyledButton(btn_frame, text="Import Multiple", command=self._on_bulk_import,
                                bg_color=AppleStyle.ACCENT_PURPLE, width=110, height=32)
        btn_import.pack(fill="x", pady=(0, 6))
        btn_import.set_tooltip("Import multiple reference images at once")
        
        btn_remove = StyledButton(btn_frame, text="Remove", command=self._on_remove_reference,
                                bg_color=AppleStyle.ACCENT_RED, width=110, height=32)
        btn_remove.pack(fill="x")
        btn_remove.set_tooltip("Remove selected reference")

    def _create_references_tab(self):
        """Create the references management tab with thumbnails."""
        tab = tk.Frame(self.notebook, bg=AppleStyle.BG_PRIMARY, padx=20, pady=20)
        self.notebook.add(tab, text="  References  ")
        
        top_frame = tk.Frame(tab, bg=AppleStyle.BG_PRIMARY)
        top_frame.pack(fill="x", pady=(0, 16))
        
        tk.Label(top_frame, text="Reference Image Management",
                font=AppleStyle.get_font(18, "bold"),
                fg=AppleStyle.TEXT_PRIMARY, bg=AppleStyle.BG_PRIMARY).pack(side="left")
        
        btn_frame = tk.Frame(top_frame, bg=AppleStyle.BG_PRIMARY)
        btn_frame.pack(side="right")
        
        btn_export = StyledButton(btn_frame, text="Export Report", command=self._on_export_report,
                                bg_color=AppleStyle.ACCENT_GREEN, width=130)
        btn_export.pack(side="left")
        btn_export.set_tooltip("Export analysis report as JSON")
        
        self.ref_gallery = ImageGallery(tab, "Reference Images with Match Scores")
        self.ref_gallery.pack(fill="both", expand=True)
        self._refresh_ref_gallery()

    def _create_logs_tab(self):
        """Create the logs tab."""
        tab = tk.Frame(self.notebook, bg=AppleStyle.BG_PRIMARY, padx=16, pady=16)
        self.notebook.add(tab, text="  Logs  ")
        
        header = tk.Frame(tab, bg=AppleStyle.BG_PRIMARY)
        header.pack(fill="x", pady=(0, 12))
        
        tk.Label(header, text="Activity Log", font=AppleStyle.get_font(15, "bold"),
                fg=AppleStyle.TEXT_PRIMARY, bg=AppleStyle.BG_PRIMARY).pack(side="left")
        
        btn_clear = StyledButton(header, text="Clear", command=self._on_clear_logs,
                               bg_color=AppleStyle.BORDER_COLOR, width=80)
        btn_clear.pack(side="right")
        
        self.logs_text = scrolledtext.ScrolledText(tab, font=AppleStyle.get_font(10),
                                                  bg=AppleStyle.BG_SECONDARY, fg=AppleStyle.TEXT_PRIMARY,
                                                  relief="flat", state="disabled")
        self.logs_text.pack(fill="both", expand=True)

    def _create_footer(self):
        """Create the application footer."""
        footer = tk.Frame(self.root, bg=AppleStyle.BG_TERTIARY, height=36)
        footer.pack(side="bottom", fill="x")
        footer.pack_propagate(False)
        
        tk.Label(footer, text="Human review required - No automated identification",
                font=AppleStyle.get_font(9), fg=AppleStyle.WARNING_YELLOW,
                bg=AppleStyle.BG_TERTIARY).pack(side="left", padx=16, pady=8)
        
        self.status_label = tk.Label(footer, text=f"References: {len(self.reference_ids)} | Ready",
                                     font=AppleStyle.get_font(9), fg=AppleStyle.TEXT_SECONDARY,
                                     bg=AppleStyle.BG_TERTIARY)
        self.status_label.pack(side="right", padx=16, pady=8)

    def _refresh_references(self):
        """Refresh references list."""
        self.refs_listbox.delete(0, tk.END)
        for ref_id in self.reference_ids:
            self.refs_listbox.insert(tk.END, f"  {ref_id}")
        self.refs_count_label.config(text=f"{len(self.reference_ids)} loaded")
        self.status_label.config(text=f"References: {len(self.reference_ids)} | Ready")

    def _refresh_ref_gallery(self):
        """Refresh reference gallery with thumbnails."""
        for widget in self.ref_gallery.winfo_children():
            widget.destroy()
        
        if not self.reference_ids:
            tk.Label(self.ref_gallery, text="No reference images loaded.\nImport images to get started.",
                    font=AppleStyle.get_font(12), fg=AppleStyle.TEXT_SECONDARY,
                    bg=AppleStyle.BG_PRIMARY, justify="center").pack(expand=True)
            return
        
        for ref_id in self.reference_ids:
            match_info = ""
            if self.current_embedding is not None and self.reference_embeddings:
                try:
                    idx = self.reference_ids.index(ref_id)
                    ref_emb = self.reference_embeddings[idx]
                    if ref_emb is not None:
                        sim = self.comparator.cosine_similarity(self.current_embedding, ref_emb)
                        match_info = f"{sim:.1%} match"
                except:
                    pass
            
            placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
            self.ref_gallery.add_image(placeholder, ref_id, match_info=match_info)

    def log_message(self, message: str, level: str = "info"):
        """Add message to logs."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        icons = {"info": "i", "success": "OK", "warning": "!", "error": "X"}
        icon = icons.get(level, "i")
        formatted = f"[{timestamp}] [{icon}] {message}"
        
        self.logs_text.config(state="normal")
        self.logs_text.insert(tk.END, formatted + "\n")
        self.logs_text.see(tk.END)
        self.logs_text.config(state="disabled")
        print(formatted)

    def _on_select_image(self):
        """Select image for analysis."""
        filepath = filedialog.askopenfilename(title="Select Image for Analysis",
                filetypes=(("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")))
        
        if filepath:
            try:
                image = cv2.imread(filepath)
                if image is None:
                    raise ValueError("Could not load image")
                
                self.current_image = image
                self.current_faces = []
                self.current_embedding = None
                self.source_viewer.load_image(image)
                self.faces_gallery.clear()
                
                self._clear_results()
                self.status_card.update(icon="◉", 
                                        message=f"Loaded: {os.path.basename(filepath)}\nResolution: {image.shape[1]}×{image.shape[0]}\nClick 'Detect Faces' to continue.",
                                        status="info")
                self.log_message(f"Image loaded: {os.path.basename(filepath)}", "success")
                self._refresh_ref_gallery()
            except Exception as e:
                self.log_message(f"Failed to load image: {e}", "error")
                self.status_card.update(icon="✗", message="Could not load the selected image.", status="error")
                messagebox.showerror("Error", f"Could not load the image:\n{e}")

    def _on_detect_faces(self):
        """Detect faces in current image."""
        if self.current_image is None:
            self.status_card.update(icon="⚠", message="No image selected.\nPlease select an image first.", status="warning")
            self.log_message("No image selected", "warning")
            messagebox.showwarning("No Image", "Please select an image first.")
            return
        
        self.loading_indicator.set_text("Detecting faces...")
        self.loading_indicator.start()
        self.root.update()
        
        faces = self.detector.detect_faces(self.current_image)
        self.current_faces = faces
        self.faces_gallery.clear()
        
        result_image = self.current_image.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            face_img = self.current_image[y:y+h, x:x+w]
            self.faces_gallery.add_image(face_img, f"Face {i+1}")
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_image, f"Face {i+1}", (x, y-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        self.source_viewer.load_image(result_image)
        self.loading_indicator.stop()
        
        if faces:
            face_info = "\n".join([f"  Face {i+1}: {f[2]}×{f[3]} at ({f[0]}, {f[1]})" for i, f in enumerate(faces)])
            self.status_card.update(icon="✓", 
                                    message=f"Detected {len(faces)} face(s):\n{face_info}\n\nClick 'Extract' to continue.",
                                    status="success")
            self.log_message(f"Detected {len(faces)} face(s)", "success")
        else:
            self.status_card.update(icon="⚠", 
                                   message="No faces detected.\nTry with a different image or better lighting.",
                                   status="warning")
            self.log_message("No faces detected", "warning")

    def _on_extract(self):
        """Extract features from detected faces."""
        if not self.current_faces:
            self.status_card.update(icon="⚠", message="No faces detected.\nPlease run face detection first.", status="warning")
            self.log_message("No faces detected. Run detection first.", "warning")
            messagebox.showwarning("No Faces", "Please run face detection first.")
            return
        
        self.loading_indicator.set_text("Extracting features...")
        self.loading_indicator.start()
        self.root.update()
        
        face = self.current_faces[0]
        x, y, w, h = face
        face_img = self.current_image[y:y+h, x:x+w]
        
        embedding = self.extractor.extract_embedding(face_img)
        self.loading_indicator.stop()
        
        if embedding is not None:
            self.current_embedding = embedding
            self.status_card.update(icon="✓", 
                                    message="Features extracted successfully!\n128-dimensional embedding generated.\n\nReady to compare with references.",
                                    status="success")
            self.log_message("Features extracted successfully", "success")
            self._refresh_ref_gallery()
        else:
            self.status_card.update(icon="✗", 
                                   message="Could not extract features.\nThe face may be too small or low quality.",
                                   status="error")
            self.log_message("Failed to extract features", "error")

    def _on_compare(self):
        """Compare with references."""
        if self.current_embedding is None:
            self.status_card.update(icon="⚠", message="No features extracted.\nPlease run feature extraction first.", status="warning")
            self.log_message("No features extracted. Run extraction first.", "warning")
            messagebox.showwarning("No Features", "Please extract features first.")
            return
        
        if not self.reference_ids:
            self.status_card.update(icon="⚠", message="No reference images.\nPlease add reference images first.", status="warning")
            self.log_message("No references loaded", "warning")
            messagebox.showwarning("No References", "Please add reference images first.")
            return
        
        self.loading_indicator.set_text("Comparing with references...")
        self.loading_indicator.start()
        self.root.update()
        
        results = self.comparator.compare_embeddings(self.current_embedding, self.reference_embeddings, self.reference_ids)
        self.loading_indicator.stop()
        
        self._clear_results()
        
        if results:
            for ref_id, similarity in results:
                confidence = self.comparator.get_confidence_band(similarity)
                card = ResultCard(self.results_container, ref_id, similarity, confidence)
                card.pack(fill="x", pady=4)
                level = "success" if similarity > 0.6 else "info"
                self.log_message(f"Match: {ref_id} - {similarity:.1%} ({confidence})", level)
            
            self.status_card.update(icon="✓", 
                                    message=f"Comparison complete!\n{len(results)} match(es) found above threshold.\n\nReview results below.",
                                    status="success")
        else:
            tk.Label(self.results_container, text="No matches above threshold.\nConfidence was insufficient across all references.",
                    font=AppleStyle.get_font(10), fg=AppleStyle.TEXT_SECONDARY,
                    bg=AppleStyle.BG_PRIMARY, justify="left").pack(anchor="w")
            self.log_message("No significant matches", "info")
            self.status_card.update(icon="⚠", message="No matches found.\nConfidence was insufficient.", status="warning")
        
        tk.Label(self.results_container, text="\nResults require human verification.",
                font=AppleStyle.get_font(9, "italic"), fg=AppleStyle.WARNING_YELLOW,
                bg=AppleStyle.BG_PRIMARY, justify="left").pack(anchor="w", pady=(8, 0))
        self._refresh_ref_gallery()

    def _clear_results(self):
        """Clear results container."""
        for widget in self.results_container.winfo_children():
            widget.destroy()
        self.loading_indicator = LoadingIndicator(self.results_container, text="")
        self.loading_indicator.pack(anchor="w", pady=(8, 0))
        self.loading_indicator.stop()

    def _on_add_reference(self):
        """Add a new reference image."""
        filepath = filedialog.askopenfilename(title="Add Reference Image",
                filetypes=(("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")))
        
        if filepath:
            img = cv2.imread(filepath)
            if img is None:
                messagebox.showerror("Error", "Could not load image")
                return

            ref_id = os.path.splitext(os.path.basename(filepath))[0]

            success, embedding = self.reference_manager.add_reference_image(
                filepath, ref_id, {"source": "gui_import", "consent": True}
            )

            if success:
                self.reference_ids.append(ref_id)
                if embedding is not None:
                    self.reference_embeddings.append(embedding)
                else:
                    self.reference_embeddings.append(None)

                self._refresh_references()
                self._refresh_ref_gallery()
                self.log_message(f"Reference added: {ref_id}", "success")
                self.status_card.update(icon="✓", message=f"Reference '{ref_id}' added successfully.", status="success")
            else:
                self.status_card.update(icon="✗", message=f"Failed to add reference '{ref_id}'.", status="error")

    def _on_remove_reference(self):
        """Remove selected reference."""
        selection = self.refs_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a reference to remove.")
            return
        
        index = selection[0]
        ref_id = self.reference_ids[index]
        
        if messagebox.askyesno("Confirm", f"Remove reference '{ref_id}'?"):
            self.reference_ids.pop(index)
            self.reference_embeddings.pop(index)
            self._refresh_references()
            self._refresh_ref_gallery()
            self.log_message(f"Reference removed: {ref_id}", "info")

    def _on_bulk_import(self):
        """Bulk import references."""
        filepaths = filedialog.askopenfilenames(title="Import Reference Images",
                filetypes=(("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")))
        
        if filepaths:
            count = 0
            for filepath in filepaths:
                ref_id = os.path.splitext(os.path.basename(filepath))[0]
                if ref_id not in self.reference_ids:
                    success, embedding = self.reference_manager.add_reference_image(
                        filepath, ref_id, {"source": "gui_bulk_import", "consent": True}
                    )
                    self.reference_ids.append(ref_id)
                    if embedding is not None:
                        self.reference_embeddings.append(embedding)
                    else:
                        self.reference_embeddings.append(None)
                    count += 1

            self._refresh_references()
            self._refresh_ref_gallery()
            self.log_message(f"Imported {count} reference(s)", "success")
            self.status_card.update(icon="✓", message=f"Imported {count} reference(s).", status="success")

    def _on_export_report(self):
        """Export analysis report."""
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "references_count": len(self.reference_ids),
            "references": self.reference_ids,
            "current_image_loaded": self.current_image is not None,
            "faces_detected": len(self.current_faces),
            "embedding_extracted": self.current_embedding is not None,
            "log": self.logs_text.get(1.0, tk.END)
        }
        
        filepath = filedialog.asksaveasfilename(defaultextension=".json",
                filetypes=[("JSON", "*.json"), ("Text", "*.txt")],
                initialfile=f"report_{datetime.date.today()}")
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            self.log_message(f"Report exported: {filepath}")
            messagebox.showinfo("Export Complete", f"Report saved to:\n{filepath}")

    def _on_clear_logs(self):
        """Clear the logs."""
        self.logs_text.config(state="normal")
        self.logs_text.delete(1.0, tk.END)
        self.logs_text.config(state="disabled")
        self.log_message("Logs cleared", "info")

    def _on_close(self):
        """Handle window close event."""
        self.webcam_active = False
        if self.webcam_cap:
            self.webcam_cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


def main():
    """Main entry point."""
    root = tk.Tk()
    
    if sys.platform == "darwin":
        try:
            root.createcommand('tk::mac::Quit', root.quit)
        except:
            pass
    
    app = FacialAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
