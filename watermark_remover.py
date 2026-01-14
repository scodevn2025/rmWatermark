# -*- coding: utf-8 -*-
"""
Watermark Remover - Simple & Easy
Chọn vùng bằng chuột → Xóa ngay!
Uses LaMa (Large Mask Inpainting) deep learning model for high-quality results.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw
import threading

# LaMa Deep Learning Model
try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
    simple_lama = SimpleLama()  # Initialize once
    print("✅ LaMa model loaded successfully!")
except ImportError:
    LAMA_AVAILABLE = False
    simple_lama = None
    print("⚠️ LaMa not available, falling back to OpenCV inpainting")


class WatermarkRemover:
    def __init__(self, root):
        self.root = root
        self.root.title("Watermark Remover - Easy & Simple")
        self.root.geometry("1200x650")
        self.root.configure(bg='#f5f5f5')

        # Data
        self.image_files = []
        self.current_index = 0
        self.original_image = None
        self.result_image = None
        self.selected_region = None  # (x, y, w, h)
        self.original_scale = 1.0
        self.result_scale = 1.0

        # Mouse selection
        self.selecting = False
        self.start_point = None
        self.rect_id = None

        # Settings
        self.auto_mode = tk.BooleanVar(value=True)
        self.inpaint_radius = tk.IntVar(value=20)

        self.setup_ui()
        
        # Auto-load images from input folder on startup
        # Wait 500ms for UI to fully render before loading
        self.root.after(500, self.auto_load_input_folder)

    def auto_load_input_folder(self):
        """Automatically load images from 'input' folder on startup"""
        input_dir = Path("input")
        if input_dir.exists() and input_dir.is_dir():
            unique_files = set()
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                unique_files.update(str(f) for f in input_dir.glob(ext))
                unique_files.update(str(f) for f in input_dir.glob(ext.upper()))
            
            if unique_files:
                self.image_files = sorted(list(unique_files))
                self.current_index = 0
                self.file_label.config(text=f"Đã load {len(self.image_files)} ảnh từ input/", fg='#4CAF50')
                self.load_image()
            else:
                self.file_label.config(text="Thư mục input/ trống", fg='#FF9800')

    def setup_ui(self):
        """Setup UI"""

        # Header
        header = tk.Frame(self.root, bg='#2196F3', height=50)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="🎨 WATERMARK REMOVER",
            font=('Arial', 16, 'bold'),
            bg='#2196F3',
            fg='white'
        ).pack(side=tk.LEFT, padx=20, pady=10)

        # Main area
        main = tk.Frame(self.root, bg='#f5f5f5')
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: Controls
        left = tk.Frame(main, bg='white', width=250, relief=tk.RAISED, bd=2)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        self.setup_controls(left)

        # Right: Preview
        right = tk.Frame(main, bg='#f5f5f5')
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.setup_preview(right)

    def setup_controls(self, parent):
        """Setup controls"""

        # File selection
        tk.Label(
            parent,
            text="📁 Chọn Ảnh",
            font=('Arial', 11, 'bold'),
            bg='white'
        ).pack(pady=(10, 5))

        tk.Button(
            parent,
            text="Chọn File",
            command=self.select_files,
            bg='#2196F3',
            fg='white',
            font=('Arial', 9, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            pady=5
        ).pack(fill=tk.X, padx=15, pady=3)

        tk.Button(
            parent,
            text="Chọn Thư Mục",
            command=self.select_folder,
            bg='#2196F3',
            fg='white',
            font=('Arial', 9, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            pady=5
        ).pack(fill=tk.X, padx=15, pady=3)

        self.file_label = tk.Label(
            parent,
            text="Chưa chọn ảnh",
            font=('Arial', 9),
            bg='white',
            fg='gray'
        )
        self.file_label.pack(pady=5)

        # Separator
        tk.Frame(parent, height=1, bg='#ddd').pack(fill=tk.X, padx=15, pady=8)

        # Mode selection
        tk.Label(
            parent,
            text="⚙️ Vùng Watermark",
            font=('Arial', 11, 'bold'),
            bg='white'
        ).pack(pady=(5, 5))

        tk.Radiobutton(
            parent,
            text="🤖 Tự động (góc trên trái)",
            variable=self.auto_mode,
            value=True,
            bg='white',
            font=('Arial', 9),
            command=self.on_mode_change
        ).pack(anchor=tk.W, padx=15)

        tk.Radiobutton(
            parent,
            text="🖱️ Chọn bằng chuột",
            variable=self.auto_mode,
            value=False,
            bg='white',
            font=('Arial', 9),
            command=self.on_mode_change
        ).pack(anchor=tk.W, padx=15)

        self.instruction_label = tk.Label(
            parent,
            text="",
            font=('Arial', 8),
            bg='white',
            fg='#FF5722',
            wraplength=220
        )
        self.instruction_label.pack(pady=5, padx=15)

        # Separator
        tk.Frame(parent, height=1, bg='#ddd').pack(fill=tk.X, padx=15, pady=8)

        # Parameters
        tk.Label(
            parent,
            text="🎛️ Tham Số",
            font=('Arial', 11, 'bold'),
            bg='white'
        ).pack(pady=(5, 5))

        tk.Label(parent, text="Độ Mạnh:", bg='white', font=('Arial', 8)).pack(anchor=tk.W, padx=15)
        tk.Scale(
            parent,
            from_=1, to=30,
            orient=tk.HORIZONTAL,
            variable=self.inpaint_radius,
            bg='white',
            length=210
        ).pack(padx=15)

        # Separator
        tk.Frame(parent, height=1, bg='#ddd').pack(fill=tk.X, padx=15, pady=8)

        # --- New Watermark Section ---
        tk.Label(
            parent,
            text="🆕 Thêm Watermark",
            font=('Arial', 11, 'bold'),
            bg='white'
        ).pack(pady=(5, 5))

        # Logo Selection
        logo_frame = tk.Frame(parent, bg='white')
        logo_frame.pack(fill=tk.X, padx=15)
        
        self.new_logo_path = None
        self.logo_btn = tk.Button(
            logo_frame, 
            text="📂 Chọn Logo", 
            command=self.select_logo,
            bg='#EEEEEE',
            font=('Arial', 9)
        )
        self.logo_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.clear_logo_btn = tk.Button(
            logo_frame,
            text="❌",
            command=self.clear_logo,
            bg='#FFCDD2',
            font=('Arial', 9),
            state='disabled'
        )
        self.clear_logo_btn.pack(side=tk.LEFT, padx=(5, 0))

        # Position
        self.wm_position = tk.StringVar(value="Góc Trái Trên")
        positions = ["Góc Trái Trên", "Góc Phải Trên", "Góc Trái Dưới", "Góc Phải Dưới", "Chính Giữa"]
        tk.OptionMenu(parent, self.wm_position, *positions).pack(fill=tk.X, padx=15, pady=5)

        # Parameters (Scale & Opacity)
        params_f = tk.Frame(parent, bg='white')
        params_f.pack(fill=tk.X, padx=15)

        tk.Scale(
            params_f, from_=10, to=100, orient=tk.HORIZONTAL, 
            label="Kích thước (%)", bg='white', length=95
        ).pack(side=tk.LEFT)
        # Wait, I need to bind variables. Let's create them in init or bind later.
        # I'll create variables now.
        self.wm_scale_val = tk.IntVar(value=5)
        self.wm_opacity_val = tk.IntVar(value=51) 

        # Re-creating scale with variable
        for widget in params_f.winfo_children(): widget.destroy()
        
        tk.Scale(
            params_f, from_=5, to=80, orient=tk.HORIZONTAL, 
            variable=self.wm_scale_val, label="Scale %", bg='white', length=95
        ).pack(side=tk.LEFT)
        
        tk.Scale(
            params_f, from_=10, to=100, orient=tk.HORIZONTAL,
            variable=self.wm_opacity_val, label="Opacity %", bg='white', length=95
        ).pack(side=tk.RIGHT)

        # Advanced Options
        opts_f = tk.Frame(parent, bg='white')
        opts_f.pack(fill=tk.X, padx=15, pady=2)
        
        self.wm_tiled = tk.BooleanVar(value=False)
        tk.Checkbutton(opts_f, text="Lặp lại", variable=self.wm_tiled, bg='white', font=('Arial', 8)).pack(side=tk.LEFT)
        
        self.wm_remove_bg = tk.BooleanVar(value=True)
        tk.Checkbutton(opts_f, text="Xóa nền trắng", variable=self.wm_remove_bg, bg='white', font=('Arial', 8)).pack(side=tk.LEFT, padx=5)

        # Rotation
        tk.Label(opts_f, text="Xoay:", bg='white', font=('Arial', 8)).pack(side=tk.LEFT, padx=(5,0))
        self.wm_angle = tk.IntVar(value=0)
        tk.Spinbox(opts_f, from_=-180, to=180, textvariable=self.wm_angle, width=4, font=('Arial', 8)).pack(side=tk.LEFT)

        # Separator
        tk.Frame(parent, height=1, bg='#ddd').pack(fill=tk.X, padx=15, pady=8)

        # Action buttons
        tk.Button(
            parent,
            text="✨ XÓA WATERMARK",
            command=self.remove_watermark,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            pady=8
        ).pack(fill=tk.X, padx=15, pady=5)

        tk.Button(
            parent,
            text="💾 Lưu",
            command=self.save_image,
            bg='#FF9800',
            fg='white',
            font=('Arial', 9, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            pady=6
        ).pack(fill=tk.X, padx=15, pady=3)

        tk.Button(
            parent,
            text="▶️ Hàng Loạt",
            command=self.batch_process,
            bg='#9C27B0',
            fg='white',
            font=('Arial', 9, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            pady=6
        ).pack(fill=tk.X, padx=15, pady=3)

        self.on_mode_change()

    def setup_preview(self, parent):
        """Setup preview area"""

        # Navigation
        nav = tk.Frame(parent, bg='#f5f5f5')
        nav.pack(fill=tk.X, pady=(0, 10))

        tk.Button(
            nav,
            text="◀️ Trước",
            command=self.prev_image,
            bg='#607D8B',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=5
        ).pack(side=tk.LEFT, padx=5)

        self.nav_label = tk.Label(
            nav,
            text="",
            font=('Arial', 11, 'bold'),
            bg='#f5f5f5'
        )
        self.nav_label.pack(side=tk.LEFT, padx=20)

        tk.Button(
            nav,
            text="Sau ▶️",
            command=self.next_image,
            bg='#607D8B',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=5
        ).pack(side=tk.LEFT, padx=5)

        # Preview container
        preview = tk.Frame(parent, bg='#f5f5f5')
        preview.pack(fill=tk.BOTH, expand=True)

        # Original
        original_frame = tk.LabelFrame(
            preview,
            text="Ảnh Gốc",
            font=('Arial', 12, 'bold'),
            bg='white',
            relief=tk.RAISED,
            bd=2
        )
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.original_canvas = tk.Canvas(
            original_frame,
            bg='#f0f0f0',
            highlightthickness=0
        )
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bind mouse events
        self.original_canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.original_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.original_canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        # Result
        result_frame = tk.LabelFrame(
            preview,
            text="Kết Quả",
            font=('Arial', 12, 'bold'),
            bg='white',
            relief=tk.RAISED,
            bd=2
        )
        result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.result_canvas = tk.Canvas(
            result_frame,
            bg='#f0f0f0',
            highlightthickness=0
        )
        self.result_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Result Navigation
        res_nav = tk.Frame(result_frame, bg='white')
        res_nav.pack(fill=tk.X, pady=(0, 5))
        
        tk.Button(
            res_nav, text="◀️ Trước", command=self.prev_result,
            bg='#607D8B', fg='white', relief=tk.FLAT, font=('Arial', 9)
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            res_nav, text="Sau ▶️", command=self.next_result,
            bg='#607D8B', fg='white', relief=tk.FLAT, font=('Arial', 9)
        ).pack(side=tk.RIGHT, padx=10)

        # Progress
        self.progress_label = tk.Label(
            parent,
            text="",
            font=('Arial', 10, 'bold'),
            bg='#f5f5f5',
            fg='#4CAF50'
        )
        self.progress_label.pack(pady=5)

    def on_mode_change(self):
        """Change mode"""
        if self.auto_mode.get():
            self.instruction_label.config(text="Chế độ tự động sẽ xóa watermark ở góc trên trái")
            self.selected_region = None
            if hasattr(self, 'original_canvas'):
                self.original_canvas.config(cursor="")
        else:
            self.instruction_label.config(text="⚠️ Click và kéo chuột trên ảnh gốc để chọn vùng watermark")
            if hasattr(self, 'original_canvas'):
                self.original_canvas.config(cursor="crosshair")

    def select_files(self):
        """Select files"""
        files = filedialog.askopenfilenames(
            title="Chọn ảnh",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if files:
            self.image_files = list(files)
            self.current_index = 0
            self.file_label.config(text=f"Đã chọn {len(self.image_files)} ảnh", fg='#4CAF50')
            self.load_image()

    def select_folder(self):
        """Select folder"""
        folder = filedialog.askdirectory(title="Chọn thư mục")
        if folder:
            folder_path = Path(folder)
            folder_path = Path(folder)
            unique_files = set()
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                # On Windows, glob is case-insensitive, so *.jpg might match *.JPG
                # We use a set to avoid duplicates
                unique_files.update(str(f) for f in folder_path.glob(ext))
                unique_files.update(str(f) for f in folder_path.glob(ext.upper()))
            
            self.image_files = sorted(list(unique_files))
            self.current_index = 0
            self.file_label.config(text=f"Đã chọn {len(self.image_files)} ảnh", fg='#4CAF50')
            self.load_image()

    def prev_image(self):
        """Previous image"""
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def next_image(self):
        """Next image"""
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_image()

    # --- New Watermark Logic ---
    def select_logo(self):
        path = filedialog.askopenfilename(
            title="Chọn Logo Watermark",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        if path:
            self.new_logo_path = path
            self.logo_btn.config(text=f"✅ {Path(path).name}", bg='#C8E6C9')
            self.clear_logo_btn.config(state='normal')
            
    def clear_logo(self):
        self.new_logo_path = None
        self.logo_btn.config(text="📂 Chọn Logo", bg='#EEEEEE')
        self.clear_logo_btn.config(state='disabled')
        
    def _apply_new_watermark(self, image):
        """Apply new logo watermark to image"""
        if not self.new_logo_path:
            return image
            
        try:
            # Load logo
            logo = Image.open(self.new_logo_path).convert("RGBA")
            h_img, w_img = image.shape[:2]
            
            # 1. Remove White Background (if selected)
            if self.wm_remove_bg.get():
                datas = logo.getdata()
                new_data = []
                for item in datas:
                    # Change all white (also shades of whites) to transparent
                    if item[0] > 200 and item[1] > 200 and item[2] > 200:
                        new_data.append((255, 255, 255, 0))
                    else:
                        new_data.append(item)
                logo.putdata(new_data)
                
            # 2. Rotation
            angle = self.wm_angle.get()
            if angle != 0:
                logo = logo.rotate(angle, expand=True, resample=Image.BICUBIC)
            
            # Calculate size
            scale = self.wm_scale_val.get() / 100.0
            
            # Resize logo maintaining aspect ratio
            logo_w, logo_h = logo.size
            aspect = logo_w / logo_h
            
            # Target width based on image width
            target_w = int(w_img * scale)
            target_h = int(target_w / aspect)
            
            if target_w <= 0 or target_h <= 0: return image
            
            logo = logo.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
            # Apply opacity
            alpha = logo.split()[3]
            opacity = self.wm_opacity_val.get() / 100.0
            alpha = alpha.point(lambda p: int(p * opacity))
            logo.putalpha(alpha)
            
            # Create overlay
            overlay = Image.new('RGBA', (w_img, h_img), (0, 0, 0, 0))
            
            # 3. Tiling (Repeated Pattern)
            if self.wm_tiled.get():
                # Spacing
                space_x = int(target_w * 0.5)
                space_y = int(target_h * 0.5)
                
                for y in range(0, h_img, target_h + space_y):
                    # Stagger rows for better look? (Optional, let's keep simple grid for now)
                    offset = 0 if (y // (target_h + space_y)) % 2 == 0 else int(target_w/2)
                    
                    for x in range(-int(target_w/2), w_img, target_w + space_x):
                        overlay.paste(logo, (x + offset, y), logo)
                        
            else:
                # Normal Positioning
                pos = self.wm_position.get()
                padding = int(w_img * 0.02) # 2% padding
                
                if pos == "Góc Trái Trên":
                    x, y = padding, padding
                elif pos == "Góc Phải Trên":
                    x, y = w_img - target_w - padding, padding
                elif pos == "Góc Trái Dưới":
                    x, y = padding, h_img - target_h - padding
                elif pos == "Góc Phải Dưới":
                    x, y = w_img - target_w - padding, h_img - target_h - padding
                else: # Center
                    x, y = (w_img - target_w) // 2, (h_img - target_h) // 2
                    
                overlay.paste(logo, (x, y), logo)
            
            # Composite
            base_img = Image.fromarray(image)
            base_img.paste(overlay, (0, 0), overlay)
            
            return np.array(base_img)
            
        except Exception as e:
            print(f"Error applying watermark: {e}")
            return image

    # --- Result Navigation ---
    def _refresh_output_files(self):
        """Scan output folder for images"""
        output_dir = Path("output")
        if not output_dir.exists():
            self.output_files = []
            return
            
        unique_files = set()
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            unique_files.update(str(f) for f in output_dir.glob(ext))
            unique_files.update(str(f) for f in output_dir.glob(ext.upper()))
            
        self.output_files = sorted(list(unique_files))

    def prev_result(self):
        """Show previous result from output folder"""
        self._refresh_output_files()
        if not hasattr(self, 'current_output_index'): self.current_output_index = 0
            
        if self.output_files:
            self.current_output_index = (self.current_output_index - 1) % len(self.output_files)
            self._display_result_file()

    def next_result(self):
        """Show next result from output folder"""
        self._refresh_output_files()
        if not hasattr(self, 'current_output_index'): self.current_output_index = 0
        
        if self.output_files:
            self.current_output_index = (self.current_output_index + 1) % len(self.output_files)
            self._display_result_file()
            
    def _display_result_file(self):
        if not self.output_files: return
        
        path = self.output_files[self.current_output_index]
        try:
             # Load with encoding support
            stream = open(path, "rb")
            bytes_data = bytearray(stream.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
            
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.display_image(image_rgb, self.result_canvas, is_original=False)
        except Exception as e:
            print(f"Error loading result: {e}")

    def load_image(self):
        """Load image"""
        if not self.image_files:
            return

        try:
            image_path = self.image_files[self.current_index]
            self.original_image = cv2.imread(image_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

            self.display_image(self.original_image, self.original_canvas, is_original=True)
            self.nav_label.config(text=f"Ảnh {self.current_index + 1}/{len(self.image_files)}")

            # Clear result
            self.result_canvas.delete("all")
            self.result_image = None
            self.selected_region = None

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể load ảnh: {str(e)}")

    def display_image(self, image, canvas, is_original=False):
        """Display image"""
        if image is None:
            return

        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        # Use larger default sizes for better initial display
        if canvas_width <= 1:
            canvas_width = 450
        if canvas_height <= 1:
            canvas_height = 500
        
        # Force update canvas to get actual size
        canvas.update_idletasks()

        h, w = image.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Calculate offset for centering
        offset_x = (canvas_width - new_w) // 2
        offset_y = (canvas_height - new_h) // 2

        if is_original:
            self.original_scale = scale
            # Store offset and display size for mouse coordinate conversion
            self.original_offset_x = offset_x
            self.original_offset_y = offset_y
            self.original_display_w = new_w
            self.original_display_h = new_h
        else:
            self.result_scale = scale

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil_image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_image)

        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER, tags="image")
        canvas.image = photo

    def show_loading(self):
        """Show loading overlay"""
        if not hasattr(self, 'loading_text_id') or self.loading_text_id is None:
            w = self.original_canvas.winfo_width()
            h = self.original_canvas.winfo_height()
            self.loading_text_id = self.original_canvas.create_text(
                w // 2, h // 2,
                text="⏳ Đang xử lý...", 
                font=('Arial', 16, 'bold'),
                fill='#FF5722',
                tags="loading"
            )
            self.original_canvas.config(state='disabled')

    def hide_loading(self):
        """Hide loading overlay"""
        if hasattr(self, 'loading_text_id') and self.loading_text_id:
            self.original_canvas.delete(self.loading_text_id)
            self.loading_text_id = None
            self.original_canvas.config(state='normal')

    def on_mouse_press(self, event):
        """Mouse press - start selection"""
        if self.auto_mode.get() or self.original_image is None:
            return

        self.selecting = True
        self.start_point = (event.x, event.y)

        # Clear previous selection visuals
        self.original_canvas.delete("selection_overlay")

    def on_mouse_drag(self, event):
        """Mouse drag - visual feedback with dimming"""
        if not self.selecting:
            return

        self.original_canvas.delete("selection_overlay")

        x1, y1 = self.start_point
        x2, y2 = event.x, event.y
        
        # Normalize coordinates
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        w_canvas = self.original_canvas.winfo_width()
        h_canvas = self.original_canvas.winfo_height()
        
        # 1. Draw Dimming Overlay (4 rectangles around selection)
        # Use stipple='gray50' for transparency effect
        # Top
        self.original_canvas.create_rectangle(0, 0, w_canvas, y_min, fill='black', stipple='gray50', outline='', tags="selection_overlay")
        # Bottom
        self.original_canvas.create_rectangle(0, y_max, w_canvas, h_canvas, fill='black', stipple='gray50', outline='', tags="selection_overlay")
        # Left
        self.original_canvas.create_rectangle(0, y_min, x_min, y_max, fill='black', stipple='gray50', outline='', tags="selection_overlay")
        # Right
        self.original_canvas.create_rectangle(x_max, y_min, w_canvas, y_max, fill='black', stipple='gray50', outline='', tags="selection_overlay")
        
        # 2. Main Selection Box
        self.original_canvas.create_rectangle(
            x_min, y_min, 
            x_max, y_max,
            outline='#00FF00', # Green feels more "confirm"
            width=2,
            dash=(4, 4),        # Dashed line for dynamic feel
            tags="selection_overlay"
        )
        
        # 3. Size Tooltip
        text = f"{x_max-x_min}x{y_max-y_min}"
        self.original_canvas.create_text(
            x_max + 10, y_max + 10, 
            text=text, 
            fill='#00FF00', 
            font=('Arial', 10, 'bold'), 
            anchor=tk.NW,
            tags="selection_overlay"
        )

    def on_mouse_release(self, event):
        """Mouse release"""
        if not self.selecting:
            return

        self.selecting = False
        end_point = (event.x, event.y)

        # Convert to image coordinates accounting for centering offset
        h_orig, w_orig = self.original_image.shape[:2]
        
        # Use stored offset and scale from display_image (NOT recalculated!)
        scale = self.original_scale
        offset_x = getattr(self, 'original_offset_x', 0)
        offset_y = getattr(self, 'original_offset_y', 0)

        # Debug print
        print(f"Canvas click: start={self.start_point}, end={end_point}")
        print(f"Image size: {w_orig}x{h_orig}, scale={scale:.2f}, offset=({offset_x},{offset_y})")

        # Convert canvas coords to image coords
        x1 = int((min(self.start_point[0], end_point[0]) - offset_x) / scale)
        y1 = int((min(self.start_point[1], end_point[1]) - offset_y) / scale)
        x2 = int((max(self.start_point[0], end_point[0]) - offset_x) / scale)
        y2 = int((max(self.start_point[1], end_point[1]) - offset_y) / scale)

        print(f"Before clamp: ({x1},{y1}) to ({x2},{y2})")

        # Clamp to image bounds
        x1 = max(0, min(x1, w_orig - 1))
        y1 = max(0, min(y1, h_orig - 1))
        x2 = max(0, min(x2, w_orig))
        y2 = max(0, min(y2, h_orig))

        # Ensure x2 > x1 and y2 > y1
        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1

        width = x2 - x1
        height = y2 - y1

        print(f"After clamp: ({x1},{y1}) size {width}x{height}")

        self.selected_region = (x1, y1, width, height)
        # Notify user of successful selection area
        self.root.title(f"Watermark Remover - Vùng đã chọn: {width}x{height}px")

    def remove_watermark(self):
        """Remove watermark"""
        if self.original_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return

        if not self.auto_mode.get() and self.selected_region is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn vùng watermark bằng chuột!")
            return

        self.progress_label.config(text="⏳ Đang xóa watermark...")
        self.show_loading()
        self.root.update()

        thread = threading.Thread(target=self._process_thread)
        thread.start()

    def _process_thread(self):
        """Process thread"""
        try:
            result = self._remove_watermark_from_image(self.original_image)
            
            # Apply new watermark if selected
            result = self._apply_new_watermark(result)
            
            self.result_image = result

            self.root.after(0, lambda: self.display_image(result, self.result_canvas, is_original=False))
            self.root.after(0, lambda: self.progress_label.config(text="✅ Hoàn thành!"))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
        finally:
            self.root.after(0, self.hide_loading)
            self.root.after(0, self.hide_loading)
            self.root.after(0, lambda: self.root.after(2000, lambda: self.progress_label.config(text="")))

    def _detect_watermark_bounds(self, image):
        """
        Smart heuristic to detect 'MI VIETNAM.VN' style watermarks (white text).
        Returns: (success_bool, (x, y, w, h))
        """
        h, w = image.shape[:2]
        
        # Focus on top-left quadrant where logo typically is
        # Scan slightly wider area: 40% width, 15% height
        scan_w = int(w * 0.4)
        scan_h = int(h * 0.15)
        
        roi = image[0:scan_h, 0:scan_w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # 1. High-Pass Filter or Adaptive Threshold to find text edges
        # The watermark is usually white text with some shadow or contrast
        # Use Morphological Gradient to find edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Threshold to get strong edges
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # 2. Connect horizontal components (letters -> words)
        # Use a wide kernel to connect "M I V I E T..."
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, connect_kernel)
        
        # 3. Find contours
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        possible_regions = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            
            # Filter noise
            if cw < 30 or ch < 10: continue
            
            # Aspect ratio check: Watermark is usually wide (text)
            aspect = cw / float(ch)
            if aspect < 2.0: continue # Likely not our long text URL
            
            possible_regions.append((x, y, cw, ch))
            
        if not possible_regions:
            return False, (0, 0, 0, 0)
            
        # 4. Merge regions (in case "MI" and "VIETNAM" are separated)
        # Find the bounding box of all valid regions
        min_x = min(r[0] for r in possible_regions)
        min_y = min(r[1] for r in possible_regions)
        max_x = max(r[0] + r[2] for r in possible_regions)
        max_y = max(r[1] + r[3] for r in possible_regions)
        
        # Pad the result slightly
        pad_x = 10
        pad_y = 5
        
        final_x = max(0, min_x - pad_x)
        final_y = max(0, min_y - pad_y)
        final_w = (max_x - min_x) + pad_x * 2
        final_h = (max_y - min_y) + pad_y * 2
        
        # Safety check: Region shouldn't be too huge (e.g. false positive complex background)
        if final_w > scan_w * 0.9 or final_h > scan_h * 0.8:
            return False, (0, 0, 0, 0)
            
        return True, (final_x, final_y, final_w, final_h)

    def _remove_watermark_from_image(self, image):
        """
        LaMa Deep Learning Watermark Removal.
        Uses mirror padding for boundary safety and aggressive dilation.
        """
        h, w = image.shape[:2]

        # Determine watermark region
        if self.auto_mode.get():
            # AI / Smart Detection System
            detection_success, (dx, dy, dw, dh) = self._detect_watermark_bounds(image)
            
            if detection_success:
                x, y, wm_w, wm_h = dx, dy, dw, dh
                print(f"🎯 AI Detection matched: {wm_w}x{wm_h} at ({x},{y})")
            else:
                # Fallback to standard region if detection fails
                x, y = 0, 0
                wm_w = int(w * 0.28)
                wm_h = int(h * 0.065)
                print(f"⚠️ AI Detection failed, using fallback: {wm_w}x{wm_h} at ({x},{y})")
        else:
            x, y, wm_w, wm_h = self.selected_region
            print(f"👆 Manual selection: {wm_w}x{wm_h} at ({x},{y})")

        # Force expansion to edges if close - ONLY for auto mode
        # For manual mode, respect the exact selection
        if self.auto_mode.get():
            edge_snap = 10
            if x < edge_snap: 
                wm_w += x
                x = 0
            if y < edge_snap:
                wm_h += y
                y = 0

        print(f"🔧 Final region to remove: x={x}, y={y}, w={wm_w}, h={wm_h}")

        # Create binary mask (white = area to inpaint)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y:y+wm_h, x:x+wm_w] = 255
        
        # Dilation for coverage - lighter for manual mode to avoid affecting surrounding content
        if self.auto_mode.get():
            # Aggressive dilation for auto detection
            kernel = np.ones((9, 9), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
        else:
            # Moderate dilation for manual selection - enough to cover watermark edges
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)

        print(f"✅ Mask created with {np.sum(mask > 0)} pixels to inpaint")

        # Use LaMa if available
        if LAMA_AVAILABLE and simple_lama is not None:
            try:
                print("🚀 Using LaMa model for inpainting...")
                
                # For efficiency, crop around watermark with large context
                pad = 150  # Large padding for better context
                
                # Calculate crop bounds with padding
                crop_x1 = max(0, x - pad)
                crop_y1 = max(0, y - pad)
                crop_x2 = min(w, x + wm_w + pad)
                crop_y2 = min(h, y + wm_h + pad)
                
                print(f"📦 Crop region: ({crop_x1},{crop_y1}) to ({crop_x2},{crop_y2})")
                
                # Crop image and mask
                crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                
                print(f"📐 Crop size: {crop_img.shape}, Mask white pixels: {np.sum(crop_mask > 0)}")
                
                # Run LaMa inpainting
                pil_img = Image.fromarray(crop_img)
                pil_mask = Image.fromarray(crop_mask)
                
                print("⏳ Running LaMa inpainting...")
                result_pil = simple_lama(pil_img, pil_mask)
                result_crop = np.array(result_pil)
                print(f"✅ LaMa done! Result shape: {result_crop.shape}")
                
                # Handle size mismatch
                if result_crop.shape[:2] != crop_img.shape[:2]:
                    result_crop = cv2.resize(result_crop, (crop_img.shape[1], crop_img.shape[0]))
                
                # Create final result
                final_result = image.copy()
                
                # Calculate where the watermark region is within the crop
                wm_in_crop_x = x - crop_x1
                wm_in_crop_y = y - crop_y1
                
                # Extract just the watermark region from the inpainted result
                # Add small padding for smoother edges
                pad_blend = 5
                blend_x1 = max(0, wm_in_crop_x - pad_blend)
                blend_y1 = max(0, wm_in_crop_y - pad_blend)
                blend_x2 = min(result_crop.shape[1], wm_in_crop_x + wm_w + pad_blend)
                blend_y2 = min(result_crop.shape[0], wm_in_crop_y + wm_h + pad_blend)
                
                # Get the region to paste
                inpainted_region = result_crop[blend_y1:blend_y2, blend_x1:blend_x2]
                
                # Calculate destination coordinates
                dest_x1 = crop_x1 + blend_x1
                dest_y1 = crop_y1 + blend_y1
                dest_x2 = dest_x1 + inpainted_region.shape[1]
                dest_y2 = dest_y1 + inpainted_region.shape[0]
                
                # Paste the inpainted region
                final_result[dest_y1:dest_y2, dest_x1:dest_x2] = inpainted_region
                
                print(f"📍 Pasted region: ({dest_x1},{dest_y1}) to ({dest_x2},{dest_y2})")
                print("🎉 Watermark removal complete!")
                return final_result
                
            except Exception as e:
                print(f"❌ LaMa error: {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback to OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        radius = self.inpaint_radius.get()
        result = cv2.inpaint(image_bgr, mask, radius, cv2.INPAINT_NS)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    def _create_watermark_mask(self, image_bgr, x, y, region_w, region_h):
        """
        Create watermark mask - simplified approach for reliable removal.
        Uses full region with feathered edges for natural blending.
        """
        h, w = image_bgr.shape[:2]
        
        # Create full region mask (more reliable than detection)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Fill the watermark region completely
        mask[y:y+region_h, x:x+region_w] = 255
        
        # Create feathered edges for smooth blending (only at non-image-edge boundaries)
        feather_size = max(5, min(region_w, region_h) // 10)
        
        # Only feather interior edges, not edges touching image boundary
        roi_mask = mask[y:y+region_h, x:x+region_w].copy()
        
        # Create gradient at edges that don't touch image boundary
        for i in range(feather_size):
            alpha = int(255 * (i + 1) / feather_size)
            # Only apply feathering on edges not at image boundary
            # Bottom edge (if not at image bottom)
            if y + region_h < h:
                roi_mask[-(i+1), :] = min(roi_mask[-(i+1), 0], alpha)
            # Right edge (if not at image right)
            if x + region_w < w:
                roi_mask[:, -(i+1)] = np.minimum(roi_mask[:, -(i+1)], alpha)
        
        mask[y:y+region_h, x:x+region_w] = roi_mask
        
        return mask

    def _pyramid_inpaint(self, image, mask, x, y, region_w, region_h):
        """Multi-scale pyramid inpainting for structure preservation"""
        result = image.copy()
        radius = self.inpaint_radius.get()
        
        # Build Gaussian pyramid (3 levels)
        num_levels = 3
        img_pyramid = [result.copy()]
        mask_pyramid = [mask.copy()]
        
        for i in range(num_levels - 1):
            img_pyramid.append(cv2.pyrDown(img_pyramid[-1]))
            mask_pyramid.append(cv2.pyrDown(mask_pyramid[-1]))
            # Ensure mask stays binary
            mask_pyramid[-1] = (mask_pyramid[-1] > 127).astype(np.uint8) * 255
        
        # Inpaint from coarsest to finest level
        for level in range(num_levels - 1, -1, -1):
            img = img_pyramid[level]
            msk = mask_pyramid[level]
            
            # Use appropriate radius for each level
            level_radius = max(3, radius // (2 ** level))
            
            # Inpaint at this level
            img = cv2.inpaint(img, msk, level_radius, cv2.INPAINT_NS)
            
            # Upsample if not at finest level
            if level > 0:
                upsampled = cv2.pyrUp(img)
                # Match size to finer level
                target_size = img_pyramid[level - 1].shape[:2][::-1]
                upsampled = cv2.resize(upsampled, target_size)
                
                # Blend upsampled with original where known
                fine_mask = mask_pyramid[level - 1]
                fine_mask_3c = np.stack([fine_mask] * 3, axis=-1) / 255.0
                img_pyramid[level - 1] = (
                    upsampled * fine_mask_3c + 
                    img_pyramid[level - 1] * (1 - fine_mask_3c)
                ).astype(np.uint8)
            else:
                result = img
        
        return result

    def _texture_synthesis_refinement(self, result, original, mask, x, y, region_w, region_h):
        """Patch-based texture synthesis for natural texture restoration"""
        h, w = result.shape[:2]
        
        # Find best texture source regions
        patch_size = max(16, min(region_w, region_h) // 4)
        
        # Sample texture from surrounding areas
        source_regions = []
        
        # Below
        if y + region_h + patch_size < h:
            source_regions.append(original[y+region_h:y+region_h+patch_size*2, x:x+region_w])
        # Right
        if x + region_w + patch_size < w:
            source_regions.append(original[y:y+region_h, x+region_w:x+region_w+patch_size*2])
        # Above
        if y - patch_size*2 >= 0:
            source_regions.append(original[y-patch_size*2:y, x:x+region_w])
        # Left
        if x - patch_size*2 >= 0:
            source_regions.append(original[y:y+region_h, x-patch_size*2:x])
        
        if not source_regions:
            return result
        
        # Find source with most similar texture to inpainted region
        roi_result = result[y:y+region_h, x:x+region_w]
        
        best_source = None
        best_score = float('inf')
        
        for src in source_regions:
            if src.size == 0 or src.shape[0] < 10 or src.shape[1] < 10:
                continue
            # Compare using gradient histogram (texture measure)
            src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.cvtColor(roi_result, cv2.COLOR_BGR2GRAY)
            
            src_grad = cv2.Laplacian(src_gray, cv2.CV_64F)
            roi_grad = cv2.Laplacian(cv2.resize(roi_gray, (src_gray.shape[1], src_gray.shape[0])), cv2.CV_64F)
            
            score = np.abs(np.std(src_grad) - np.std(roi_grad))
            if score < best_score:
                best_score = score
                best_source = src
        
        if best_source is None:
            return result
        
        # Apply texture transfer using color transfer technique
        try:
            # Resize source to match ROI
            source_resized = cv2.resize(best_source, (region_w, region_h))
            
            # Convert to LAB for color matching
            roi_lab = cv2.cvtColor(roi_result, cv2.COLOR_BGR2LAB).astype(np.float32)
            src_lab = cv2.cvtColor(source_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            # Match mean and std of L channel (luminance) only for texture
            # Keep color from inpainted result
            for i in [1, 2]:  # Only a and b channels
                src_mean = np.mean(src_lab[:, :, i])
                src_std = np.std(src_lab[:, :, i])
                roi_mean = np.mean(roi_lab[:, :, i])
                roi_std = np.std(roi_lab[:, :, i])
                
                if src_std > 0:
                    roi_lab[:, :, i] = (roi_lab[:, :, i] - roi_mean) * (src_std / max(roi_std, 1)) + src_mean
            
            roi_lab = np.clip(roi_lab, 0, 255).astype(np.uint8)
            refined_roi = cv2.cvtColor(roi_lab, cv2.COLOR_LAB2BGR)
            
            # Blend refined texture with inpainted result
            # Use mask to only apply to watermark areas
            roi_mask = mask[y:y+region_h, x:x+region_w]
            roi_mask_3c = np.stack([roi_mask / 255.0] * 3, axis=-1)
            
            blended = (refined_roi * roi_mask_3c * 0.3 + roi_result * (1 - roi_mask_3c * 0.3)).astype(np.uint8)
            result[y:y+region_h, x:x+region_w] = blended
            
        except Exception:
            pass
        
        return result

    def _seamless_blend(self, result, original, x, y, region_w, region_h):
        """Final seamless blending at region boundaries"""
        h, w = result.shape[:2]
        
        # Create feathered edge mask
        edge_width = max(8, min(region_w, region_h) // 8)
        
        # Expand region slightly for blending
        bx = max(0, x - edge_width)
        by = max(0, y - edge_width)
        bw = min(w - bx, region_w + edge_width * 2)
        bh = min(h - by, region_h + edge_width * 2)
        
        # Create gradient mask
        mask = np.zeros((bh, bw), dtype=np.float32)
        
        # Calculate relative position of watermark region within blend region
        inner_x = x - bx
        inner_y = y - by
        
        # Fill inner region with 1.0
        if inner_y + region_h <= bh and inner_x + region_w <= bw:
            mask[inner_y:inner_y+region_h, inner_x:inner_x+region_w] = 1.0
        
        # Apply Gaussian blur for smooth transition
        mask = cv2.GaussianBlur(mask, (edge_width * 2 + 1, edge_width * 2 + 1), edge_width / 2)
        
        # Blend result with original at boundaries
        mask_3c = np.stack([mask] * 3, axis=-1)
        
        blend_roi_result = result[by:by+bh, bx:bx+bw]
        blend_roi_orig = original[by:by+bh, bx:bx+bw]
        
        blended = (blend_roi_result * mask_3c + blend_roi_orig * (1 - mask_3c)).astype(np.uint8)
        
        result = result.copy()
        result[by:by+bh, bx:bx+bw] = blended
        
        return result

    def save_image(self):
        """Save image to output folder"""
        if self.result_image is None:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh kết quả!")
            return

        try:
            # Create output folder if not exists
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)

            # Get current filename
            if self.image_files:
                filename = Path(self.image_files[self.current_index]).name
            else:
                filename = 'output.jpg'

            save_path = output_dir / filename

            # Save image
            image_bgr = cv2.cvtColor(self.result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), image_bgr)

            messagebox.showinfo("Thành công", f"Đã lưu vào:\noutput/{filename}")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def batch_process(self):
        """Batch process - save to output folder"""
        if not self.image_files:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return

        if not self.auto_mode.get() and self.selected_region is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn vùng watermark trước!")
            return

        # Create output folder
        output_folder = Path('output')
        output_folder.mkdir(exist_ok=True)

        thread = threading.Thread(target=self._batch_thread, args=(str(output_folder),))
        thread.start()

    def _batch_thread(self, output_folder):
        """Batch thread"""
        total = len(self.image_files)
        success = 0

        for i, image_path in enumerate(self.image_files):
            try:
                self.root.after(0, lambda idx=i, t=total: self.progress_label.config(
                    text=f"⏳ Đang xử lý: {idx + 1}/{t}"
                ))

                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Use current image as reference
                # Update UI to show current processing image
                self.current_index = i
                
                # Update display safely in main thread
                self.root.after(0, lambda img=image_rgb: self.display_image(img, self.original_canvas, is_original=True))
                self.root.after(0, lambda txt=f"Ảnh {i+1}/{total}": self.nav_label.config(text=txt))
                
                # Hack: Update original_image for processing context (if needed by modifiers)
                old_image = self.original_image
                self.original_image = image_rgb
                
                result_rgb = self._remove_watermark_from_image(image_rgb)
                
                # Apply new watermark if selected
                result_rgb = self._apply_new_watermark(result_rgb)
                
                self.original_image = old_image # Restore or keep? Actually better to update it to current if we want to support "stop" later.
                # But for now, just keep logic simple
                
                # Update result display
                self.root.after(0, lambda res=result_rgb: self.display_image(res, self.result_canvas, is_original=False))

                result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

                filename = Path(image_path).name
                output_path = Path(output_folder) / filename
                cv2.imwrite(str(output_path), result_bgr)

                success += 1

            except Exception as e:
                print(f"Error: {e}")

        self.root.after(0, lambda: self.progress_label.config(text=f"✅ Hoàn thành: {success}/{total} ảnh"))
        self.root.after(0, lambda: messagebox.showinfo(
            "Hoàn thành",
            f"Đã xử lý {success}/{total} ảnh!\n\nLưu tại: output/"
        ))


def main():
    root = tk.Tk()
    app = WatermarkRemover(root)
    root.mainloop()


if __name__ == "__main__":
    main()
