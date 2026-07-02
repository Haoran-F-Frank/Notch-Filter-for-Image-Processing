#!/usr/bin/env python3
import pathlib
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image

from bio_io.mhd_loader import AXIS_LABELS, load_mhd_slice, normalize_to_uint8, save_slice_png
from filters.notch_core import (
    FILTER_TYPE_BUTTERWORTH,
    FILTER_TYPE_GAUSSIAN,
    NotchFilterSpec,
    apply_filter_specs,
    compute_fshift,
    compute_log_magnitude,
)


class FilterPortalApp:
    def __init__(self):
        if sys.platform == "win32":
            try:
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
            except (AttributeError, OSError):
                pass

        self.root = tk.Tk()
        self.root.title("MHD Notch Filter Portal")
        self.root.geometry("1400x900")

        self.mhd_path = None
        self.slice_image = None
        self.slice_shape = None
        self.fshift_base = None
        self.log_spectrum = None
        self.filters = []
        self.selected_filter_index = None
        self.dragging = False
        self.volume_shape = None
        self._base_limits = {}
        self._axis_zoom = {}
        self._ctrl_pressed = False
        self._reload_after_id = None
        self._updating_controls = False
        self._axis_trace_ready = False
        self._last_hover_axis = None
        self._last_filtered = None
        self._large_view_window = None
        self._max_zoom = 50.0

        self._build_layout()
        self._axis_trace_ready = True
        self._create_default_filter()

    def _build_layout(self):
        toolbar = tk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        tk.Button(toolbar, text="Load MHD", command=self.load_mhd).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Close MHD", command=self.close_mhd).pack(side=tk.LEFT, padx=4)
        tk.Label(toolbar, text="Axis:").pack(side=tk.LEFT)
        self.axis_var = tk.StringVar(value="0")
        self.axis_menu = ttk.Combobox(
            toolbar,
            textvariable=self.axis_var,
            values=["0", "1", "2"],
            state="readonly",
            width=4,
        )
        self.axis_menu.pack(side=tk.LEFT, padx=4)
        self.axis_var.trace_add("write", self._on_axis_var_changed)

        tk.Button(toolbar, text="◀", width=2, command=lambda: self._step_slice(-1)).pack(side=tk.LEFT, padx=(8, 0))
        tk.Label(toolbar, text="Slice:").pack(side=tk.LEFT, padx=(4, 0))
        self.slice_var = tk.StringVar(value="0")
        self.slice_entry = tk.Entry(toolbar, textvariable=self.slice_var, width=6)
        self.slice_entry.pack(side=tk.LEFT, padx=4)
        self.slice_entry.bind("<Return>", self.on_slice_entry_commit)
        self.slice_entry.bind("<FocusOut>", self.on_slice_entry_commit)
        tk.Button(toolbar, text="▶", width=2, command=lambda: self._step_slice(1)).pack(side=tk.LEFT, padx=(0, 4))
        self.slice_max_label = tk.Label(toolbar, text="/ 0")
        self.slice_max_label.pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(toolbar, text="Load Slice", command=lambda: self.reload_slice()).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Preview", command=self.preview).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Save Filtered PNG", command=self.save_filtered).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Zoom +", command=lambda: self.zoom_by_button(True)).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Zoom -", command=lambda: self.zoom_by_button(False)).pack(side=tk.LEFT, padx=2)
        tk.Label(toolbar, text="Zoom target:").pack(side=tk.LEFT, padx=(8, 0))
        self.zoom_target_var = tk.StringVar(value="filtered")
        ttk.Combobox(
            toolbar,
            textvariable=self.zoom_target_var,
            values=["filtered", "cursor", "all", "original", "spectrum"],
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Large Filtered View", command=self.open_large_filtered_view).pack(side=tk.LEFT, padx=4)
        self.status_var = tk.StringVar(value="Load an MHD file to begin.")
        tk.Label(toolbar, textvariable=self.status_var, anchor="w").pack(side=tk.LEFT, padx=12)

        slice_bar = tk.Frame(self.root)
        slice_bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))
        tk.Label(slice_bar, text="Slice slider:").pack(side=tk.LEFT)
        self.slice_scale = tk.Scale(
            slice_bar,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            showvalue=True,
            length=500,
            command=self.on_slice_scale_changed,
        )
        self.slice_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        body = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(body)
        body.add(left, minsize=320)

        right = tk.Frame(body)
        body.add(right)

        self._build_filter_panel(left)
        self._build_plot_panel(right)

    def _build_filter_panel(self, parent):
        tk.Label(parent, text="Notch Filters", font=("Arial", 12, "bold")).pack(anchor="w", padx=8, pady=4)

        list_frame = tk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.filter_listbox = tk.Listbox(list_frame, height=12)
        self.filter_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.filter_listbox.bind("<<ListboxSelect>>", self.on_filter_selected)

        scrollbar = tk.Scrollbar(list_frame, command=self.filter_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.filter_listbox.config(yscrollcommand=scrollbar.set)

        btn_row = tk.Frame(parent)
        btn_row.pack(fill=tk.X, padx=8, pady=4)
        tk.Button(btn_row, text="Add", command=self.add_filter).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Delete", command=self.delete_filter).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Duplicate", command=self.duplicate_filter).pack(side=tk.LEFT, padx=2)

        params = tk.LabelFrame(parent, text="Selected Filter Parameters")
        params.pack(fill=tk.X, padx=8, pady=8)

        self.name_var = tk.StringVar()
        self.type_var = tk.StringVar(value=str(FILTER_TYPE_BUTTERWORTH))
        self.x_var = tk.StringVar()
        self.y_var = tk.StringVar()
        self.radius_var = tk.StringVar(value="30")
        self.order_var = tk.StringVar(value="2")
        self.enabled_var = tk.BooleanVar(value=True)

        self._add_param_row(params, "Name", tk.Entry(params, textvariable=self.name_var))
        self._add_param_row(
            params,
            "Type (0=BW, 1=Gauss)",
            ttk.Combobox(
                params,
                textvariable=self.type_var,
                values=["0", "1"],
                state="readonly",
                width=18,
            ),
        )
        self._add_param_row(params, "X", tk.Entry(params, textvariable=self.x_var, width=20))
        self._add_param_row(params, "Y", tk.Entry(params, textvariable=self.y_var, width=20))
        self._add_param_row(params, "Radius", tk.Entry(params, textvariable=self.radius_var, width=20))
        self._add_param_row(params, "Order (Butterworth)", tk.Entry(params, textvariable=self.order_var, width=20))
        tk.Checkbutton(params, text="Enabled", variable=self.enabled_var).pack(anchor="w", padx=8, pady=4)

        tk.Button(params, text="Apply Parameters", command=self.apply_parameters).pack(fill=tk.X, padx=8, pady=6)

        help_text = (
            "Tips:\n"
            "1. Use Axis to switch slice direction (0=Z/axial, 1=Y/coronal, 2=X/sagittal).\n"
            "2. Change Slice via entry, slider, ◀▶ buttons, or mouse wheel.\n"
            "3. Ctrl + mouse wheel (or Zoom +/-) to enlarge the image for tiny noise details.\n"
            "4. Set Zoom target to 'filtered' to zoom the right panel only.\n"
            "5. Use 'Large Filtered View' for a bigger popup window.\n"
            "6. Click on the spectrum to move the selected filter point.\n"
            "7. Type 0 = Butterworth, 1 = Gaussian.\n"
            "8. Press Preview to update filtered image."
        )
        tk.Label(parent, text=help_text, justify=tk.LEFT, wraplength=300).pack(anchor="w", padx=8, pady=8)

    def _add_param_row(self, parent, label, widget):
        row = tk.Frame(parent)
        row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(row, text=label, width=20, anchor="w").pack(side=tk.LEFT)
        widget.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _build_plot_panel(self, parent):
        self.plot_parent = parent
        self.fig, self.axes = plt.subplots(
            1,
            3,
            figsize=(12, 4),
            constrained_layout=True,
        )

        self.ax_original = self.axes[0]
        self.ax_spectrum = self.axes[1]
        self.ax_filtered = self.axes[2]

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.plot_parent.bind("<Configure>", self._on_plot_resize)

        self.marker_artist = None
        self.symmetric_marker = None

        self.canvas.mpl_connect("button_press_event", self.on_canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_canvas_motion)
        self.canvas.mpl_connect("button_release_event", self.on_canvas_release)
        self.canvas.mpl_connect("scroll_event", self.on_canvas_scroll)
        self.canvas.mpl_connect("axes_enter_event", self.on_axes_enter)

        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.bind("<MouseWheel>", self.on_tk_mousewheel, add="+")
        canvas_widget.bind("<Button-4>", self.on_tk_mousewheel, add="+")
        canvas_widget.bind("<Button-5>", self.on_tk_mousewheel, add="+")
        canvas_widget.bind("<Control-MouseWheel>", self.on_tk_mousewheel, add="+")
        canvas_widget.bind("<Control-Button-4>", self.on_tk_mousewheel, add="+")
        canvas_widget.bind("<Control-Button-5>", self.on_tk_mousewheel, add="+")

        self.root.bind("<Control_L>", self._on_ctrl_press, add="+")
        self.root.bind("<Control_R>", self._on_ctrl_press, add="+")
        self.root.bind("<KeyRelease-Control_L>", self._on_ctrl_release, add="+")
        self.root.bind("<KeyRelease-Control_R>", self._on_ctrl_release, add="+")

    def on_axes_enter(self, event):
        if event.inaxes in (self.ax_original, self.ax_spectrum, self.ax_filtered):
            self._last_hover_axis = event.inaxes

    def _on_ctrl_press(self, _event=None):
        self._ctrl_pressed = True

    def _on_ctrl_release(self, _event=None):
        self._ctrl_pressed = False

    def _ctrl_is_pressed(self, event=None):
        if self._ctrl_pressed:
            return True
        if event is not None:
            state = getattr(event, "state", 0)
            if state & 0x0004 or state & 0x20000:
                return True
            key = getattr(event, "key", None)
            if key and "control" in str(key).lower():
                return True
        return False

    def _scroll_direction(self, event):
        if hasattr(event, "delta") and event.delta:
            return 1 if event.delta > 0 else -1
        if hasattr(event, "num") and event.num in (4, 5):
            return -1 if event.num == 4 else 1
        step = getattr(event, "step", 0)
        if step:
            return 1 if step > 0 else -1
        return 0

    def on_canvas_scroll(self, event):
        if event.inaxes not in (self.ax_original, self.ax_spectrum, self.ax_filtered):
            return
        direction = self._scroll_direction(event)
        if direction == 0:
            return
        if self._ctrl_is_pressed(event):
            self._zoom_target_axes(event.inaxes, direction, event.xdata, event.ydata)
        else:
            self._change_slice(direction)

    def on_tk_mousewheel(self, event):
        direction = self._scroll_direction(event)
        if direction == 0:
            return
        if self._ctrl_is_pressed(event):
            axis = self._axis_under_pointer(event)
            if axis is None:
                return
            xdata, ydata = self._data_coords_from_tk_event(axis, event)
            self._zoom_target_axes(axis, direction, xdata, ydata)
        else:
            self._change_slice(direction)

    def _axis_under_pointer(self, event):
        canvas_widget = self.canvas.get_tk_widget()
        width = canvas_widget.winfo_width()
        height = canvas_widget.winfo_height()
        if width <= 0 or height <= 0:
            return None
        rel_x = event.x / width
        for axis in (self.ax_original, self.ax_spectrum, self.ax_filtered):
            bbox = axis.get_position()
            if bbox.x0 <= rel_x <= bbox.x1:
                return axis
        return self.ax_original

    def _data_coords_from_tk_event(self, axis, event):
        canvas_widget = self.canvas.get_tk_widget()
        width = canvas_widget.winfo_width()
        height = canvas_widget.winfo_height()
        if width <= 0 or height <= 0:
            return None, None
        bbox = axis.get_position()
        rel_x = (event.x / width - bbox.x0) / max(bbox.width, 1e-8)
        rel_y = (event.y / height - bbox.y0) / max(bbox.height, 1e-8)
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        xdata = xlim[0] + rel_x * (xlim[1] - xlim[0])
        ydata = ylim[0] + (1 - rel_y) * (ylim[1] - ylim[0])
        return xdata, ydata

    def _current_axis(self):
        return int(self.axis_var.get())

    def _max_slice_index(self):
        if self.volume_shape is None:
            return 0
        return max(self.volume_shape[self._current_axis()] - 1, 0)

    def _clamp_slice_index(self, slice_index):
        return max(0, min(int(slice_index), self._max_slice_index()))

    def _sync_slice_controls(self, slice_index):
        slice_index = self._clamp_slice_index(slice_index)
        max_slice = self._max_slice_index()
        self._updating_controls = True
        self.slice_var.set(str(slice_index))
        self.slice_scale.config(to=max_slice)
        self.slice_scale.set(slice_index)
        self.slice_max_label.config(text=f"/ {max_slice}")
        self._updating_controls = False
        return slice_index

    def _schedule_reload_slice(self, reset_zoom=False, new_orientation=False, delay_ms=60):
        if self._reload_after_id is not None:
            self.root.after_cancel(self._reload_after_id)

        def _run():
            self._reload_after_id = None
            self.reload_slice(reset_zoom=reset_zoom, new_orientation=new_orientation)

        self._reload_after_id = self.root.after(delay_ms, _run)

    def _step_slice(self, direction):
        if not self.mhd_path or self.volume_shape is None:
            return
        try:
            current = int(self.slice_var.get())
        except ValueError:
            current = 0
        new_slice = self._clamp_slice_index(current + direction)
        if new_slice == current:
            return
        self._sync_slice_controls(new_slice)
        self._schedule_reload_slice(reset_zoom=False)

    def on_slice_entry_commit(self, _event=None):
        if not self.mhd_path or self._updating_controls:
            return
        try:
            slice_index = self._clamp_slice_index(int(self.slice_var.get()))
        except ValueError:
            messagebox.showerror("Invalid slice", "Slice index must be an integer.")
            self._sync_slice_controls(self.slice_scale.get())
            return
        self._sync_slice_controls(slice_index)
        self.reload_slice(reset_zoom=False)

    def on_slice_scale_changed(self, value):
        if not self.mhd_path or self._updating_controls:
            return
        slice_index = self._clamp_slice_index(float(value))
        self._updating_controls = True
        self.slice_var.set(str(slice_index))
        self._updating_controls = False
        self._schedule_reload_slice(reset_zoom=False)

    def _on_axis_var_changed(self, *_args):
        if not self._axis_trace_ready or self._updating_controls:
            return
        if self.mhd_path and self.volume_shape is not None:
            self.on_axis_changed()

    def on_axis_changed(self, _event=None):
        if not self.mhd_path or self.volume_shape is None:
            return
        try:
            current = int(self.slice_var.get())
        except ValueError:
            current = 0
        slice_index = self._clamp_slice_index(current)
        self._sync_slice_controls(slice_index)
        self.reload_slice(reset_zoom=True, new_orientation=True)

    def _change_slice(self, direction):
        if not self.mhd_path or self.volume_shape is None:
            return
        self._step_slice(direction)

    def reset_zoom(self):
        self._axis_zoom = {}
        for axis in (self.ax_original, self.ax_spectrum, self.ax_filtered):
            self._restore_axis_view(axis)
        self._update_zoom_status()
        self.canvas.draw_idle()

    def _zoom_target_axes(self, cursor_axis, direction, xdata=None, ydata=None):
        target = self.zoom_target_var.get()
        zoom_in = direction > 0
        if target == "all":
            axes = [self.ax_original, self.ax_spectrum, self.ax_filtered]
        elif target == "filtered":
            axes = [self.ax_filtered]
        elif target == "original":
            axes = [self.ax_original]
        elif target == "spectrum":
            axes = [self.ax_spectrum]
        else:
            axes = [cursor_axis or self._last_hover_axis or self.ax_filtered]

        for axis in axes:
            if id(axis) not in self._base_limits:
                continue
            center = (xdata, ydata) if axis is cursor_axis else None
            self._zoom_single_axis(axis, zoom_in, center=center)
        self._update_zoom_status()
        self.canvas.draw_idle()

    def zoom_by_button(self, zoom_in):
        axis = self._last_hover_axis or self.ax_filtered
        self._zoom_target_axes(axis, 1 if zoom_in else -1)

    def _update_zoom_status(self):
        zoom = self._axis_zoom.get(id(self.ax_filtered), 1.0)
        base = self.status_var.get().split(" | zoom=")[0]
        if self.slice_image is not None:
            self.status_var.set(f"{base} | zoom={zoom:.1f}x")

    def _zoom_axes(self, target_axis, direction, xdata=None, ydata=None):
        self._zoom_target_axes(target_axis, direction, xdata, ydata)

    def _zoom_single_axis(self, axis, zoom_in, center=None):
        base_xlim, base_ylim = self._base_limits[id(axis)]
        bx0, bx1 = base_xlim
        by0, by1 = base_ylim

        current_zoom = self._axis_zoom.get(id(axis), 1.0)
        if zoom_in:
            current_zoom = min(self._max_zoom, current_zoom * 1.2)
        else:
            current_zoom = max(1.0, current_zoom / 1.2)
        self._axis_zoom[id(axis)] = current_zoom

        if current_zoom <= 1.0:
            axis.set_xlim(base_xlim)
            axis.set_ylim(base_ylim)
            return

        cx = center[0] if center and center[0] is not None else (bx0 + bx1) / 2
        cy = center[1] if center and center[1] is not None else (by0 + by1) / 2

        cur_xlim = axis.get_xlim()
        cur_ylim = axis.get_ylim()
        scale = 1 / 1.2 if zoom_in else 1.2
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale
        new_height = abs(cur_ylim[1] - cur_ylim[0]) * scale

        relx = (cx - cur_xlim[0]) / max(cur_xlim[1] - cur_xlim[0], 1e-8)
        rely = (cur_ylim[0] - cy) / max(cur_ylim[0] - cur_ylim[1], 1e-8)

        axis.set_xlim(cx - new_width * relx, cx + new_width * (1 - relx))
        axis.set_ylim(cy + new_height * (1 - rely), cy - new_height * rely)

    def _safe_remove_artist(self, artist):
        if artist is None:
            return
        try:
            if artist.axes is not None:
                artist.remove()
        except (ValueError, AttributeError, NotImplementedError):
            pass

    def _clear_markers(self):
        self._safe_remove_artist(self.marker_artist)
        self._safe_remove_artist(self.symmetric_marker)
        self.marker_artist = None
        self.symmetric_marker = None

    def _display_image(self, axis, image, title):
        if axis is self.ax_spectrum:
            self._clear_markers()
        height, width = image.shape[:2]
        axis.clear()
        axis.imshow(
            image,
            cmap="gray",
            aspect="equal",
            interpolation="nearest",
            origin="upper",
        )
        xlim = (-0.5, width - 0.5)
        ylim = (height - 0.5, -0.5)
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{title} ({width} x {height})")
        axis.set_xticks([])
        axis.set_yticks([])
        self._base_limits[id(axis)] = (xlim, ylim)
        self._restore_axis_view(axis)

    def _restore_axis_view(self, axis):
        if id(axis) not in self._base_limits:
            return
        base_xlim, base_ylim = self._base_limits[id(axis)]
        zoom = self._axis_zoom.get(id(axis), 1.0)
        if zoom <= 1.0:
            axis.set_xlim(base_xlim)
            axis.set_ylim(base_ylim)
            return
        bx0, bx1 = base_xlim
        by0, by1 = base_ylim
        cx = (bx0 + bx1) / 2
        cy = (by0 + by1) / 2
        half_w = (bx1 - bx0) / (2 * zoom)
        half_h = abs(by1 - by0) / (2 * zoom)
        axis.set_xlim(cx - half_w, cx + half_w)
        axis.set_ylim(cy + half_h, cy - half_h)

    def _resize_figure_for_image(self, height, width):
        pixels_per_inch = self.fig.dpi
        panel_width_px = max(self.plot_parent.winfo_width() - 24, 480) / 3
        panel_height_px = max(self.plot_parent.winfo_height() - 24, 240)

        image_aspect = height / width
        panel_aspect = panel_height_px / panel_width_px

        if image_aspect > panel_aspect:
            display_height = panel_height_px
            display_width = panel_height_px / image_aspect
        else:
            display_width = panel_width_px
            display_height = panel_width_px * image_aspect

        fig_width = max(display_width * 3 / pixels_per_inch, 9.0)
        fig_height = max(display_height / pixels_per_inch, 3.0)
        self.fig.set_size_inches(fig_width, fig_height, forward=True)

    def _on_plot_resize(self, _event=None):
        if self.slice_shape is None:
            return
        height, width = self.slice_shape
        self._resize_figure_for_image(height, width)
        self.canvas.draw_idle()

    def _adapt_filters_to_shape(self, height, width, previous_shape=None):
        center_x = width / 2
        center_y = height / 2

        if previous_shape is None:
            for index, spec in enumerate(self.filters):
                spec.x = center_x
                spec.y = center_y
                spec.name = spec.name or f"Filter {index + 1}"
            return

        old_height, old_width = previous_shape
        if (old_height, old_width) == (height, width):
            return

        scale_x = width / old_width
        scale_y = height / old_height
        scale_radius = (scale_x + scale_y) / 2
        for spec in self.filters:
            spec.x *= scale_x
            spec.y *= scale_y
            spec.radius *= scale_radius

    def _create_default_filter(self):
        self.filters = [
            NotchFilterSpec(
                name="Filter 1",
                filter_type=FILTER_TYPE_BUTTERWORTH,
                x=64,
                y=64,
                radius=30,
                order=2,
            )
        ]
        self.selected_filter_index = 0
        self.refresh_filter_list()
        self.load_params_to_form()

    def refresh_filter_list(self):
        self.filter_listbox.delete(0, tk.END)
        for index, spec in enumerate(self.filters):
            prefix = "*" if index == self.selected_filter_index else " "
            enabled = "ON" if spec.enabled else "OFF"
            self.filter_listbox.insert(
                tk.END,
                f"{prefix} [{enabled}] {spec.summary()}",
            )
        if self.selected_filter_index is not None:
            self.filter_listbox.selection_set(self.selected_filter_index)
            self.filter_listbox.activate(self.selected_filter_index)

    def selected_filter(self):
        if self.selected_filter_index is None:
            return None
        return self.filters[self.selected_filter_index]

    def on_filter_selected(self, _event=None):
        self.apply_parameters_silent()
        selection = self.filter_listbox.curselection()
        if not selection:
            return
        self.selected_filter_index = selection[0]
        self.load_params_to_form()
        self.refresh_filter_list()
        self.update_markers()

    def load_params_to_form(self):
        spec = self.selected_filter()
        if spec is None:
            return
        self.name_var.set(spec.name)
        self.type_var.set(str(spec.filter_type))
        self.x_var.set(f"{spec.x:.2f}")
        self.y_var.set(f"{spec.y:.2f}")
        self.radius_var.set(f"{spec.radius:.2f}")
        self.order_var.set(str(spec.order))
        self.enabled_var.set(spec.enabled)

    def apply_parameters(self):
        spec = self.selected_filter()
        if spec is None:
            return
        try:
            spec.name = self.name_var.get().strip() or spec.name
            spec.filter_type = int(self.type_var.get())
            spec.x = float(self.x_var.get())
            spec.y = float(self.y_var.get())
            spec.radius = max(float(self.radius_var.get()), 1.0)
            spec.order = max(int(float(self.order_var.get())), 1)
            spec.enabled = self.enabled_var.get()
        except ValueError as error:
            messagebox.showerror("Invalid parameter", str(error))
            return

        self.refresh_filter_list()
        self.update_markers()
        self.preview()

    def add_filter(self):
        center_x = self.slice_image.shape[1] / 2 if self.slice_image is not None else 64
        center_y = self.slice_image.shape[0] / 2 if self.slice_image is not None else 64
        new_filter = NotchFilterSpec(
            name=f"Filter {len(self.filters) + 1}",
            filter_type=FILTER_TYPE_BUTTERWORTH,
            x=center_x + 10,
            y=center_y,
            radius=30,
            order=2,
        )
        self.filters.append(new_filter)
        self.selected_filter_index = len(self.filters) - 1
        self.refresh_filter_list()
        self.load_params_to_form()
        self.update_markers()
        self.preview()

    def delete_filter(self):
        if not self.filters:
            return
        if len(self.filters) == 1:
            messagebox.showinfo("Delete filter", "At least one filter must remain.")
            return
        index = self.selected_filter_index if self.selected_filter_index is not None else 0
        del self.filters[index]
        self.selected_filter_index = min(index, len(self.filters) - 1)
        self.refresh_filter_list()
        self.load_params_to_form()
        self.update_markers()
        self.preview()

    def duplicate_filter(self):
        spec = self.selected_filter()
        if spec is None:
            return
        clone = NotchFilterSpec(
            name=f"{spec.name} copy",
            filter_type=spec.filter_type,
            x=spec.x + 5,
            y=spec.y + 5,
            radius=spec.radius,
            order=spec.order,
            enabled=spec.enabled,
        )
        self.filters.append(clone)
        self.selected_filter_index = len(self.filters) - 1
        self.refresh_filter_list()
        self.load_params_to_form()
        self.update_markers()
        self.preview()

    def load_mhd(self):
        path = filedialog.askopenfilename(
            title="Load MHD / MetaImage",
            filetypes=[
                ("MetaImage", "*.mhd *.mha *.raw"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        new_file = path != self.mhd_path
        self._clear_markers()
        self.mhd_path = path
        self._axis_zoom = {}
        self._base_limits = {}

        if new_file:
            self.slice_var.set("0")
            self.slice_shape = None
            self.volume_shape = None
            self._create_default_filter()

        self.reload_slice(reset_zoom=True, new_volume=new_file)

    def close_mhd(self):
        if self._reload_after_id is not None:
            self.root.after_cancel(self._reload_after_id)
            self._reload_after_id = None
        self._clear_markers()
        self.mhd_path = None
        self.slice_image = None
        self.slice_shape = None
        self.volume_shape = None
        self.fshift_base = None
        self.log_spectrum = None
        self._axis_zoom = {}
        self._base_limits = {}

        for axis, title in (
            (self.ax_original, "Original Slice"),
            (self.ax_spectrum, "Frequency Spectrum"),
            (self.ax_filtered, "Filtered Slice"),
        ):
            axis.clear()
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])

        self.status_var.set("MHD file closed. Click Load MHD to open another file.")
        self.canvas.draw_idle()

        self.status_var.set("MHD file closed. Click Load MHD to open another file.")
        self._sync_slice_controls(0)
        self.canvas.draw_idle()

    def reload_slice(self, reset_zoom=False, new_volume=False, new_orientation=False):
        if not self.mhd_path:
            messagebox.showinfo("Load MHD", "Please load an MHD file first.")
            return
        try:
            if reset_zoom:
                self._axis_zoom = {}
            slice_index = self._clamp_slice_index(int(self.slice_var.get()))
            axis = self._current_axis()
            if new_volume or new_orientation:
                previous_shape = None
            else:
                previous_shape = self.slice_shape
            self.slice_image, _, info = load_mhd_slice(
                self.mhd_path,
                slice_index=slice_index,
                axis=axis,
            )
            self.volume_shape = info["shape"]
            self.slice_shape = self.slice_image.shape
            height, width = self.slice_shape
            self._sync_slice_controls(slice_index)

            self.fshift_base = compute_fshift(self.slice_image)
            self.log_spectrum = compute_log_magnitude(self.fshift_base)

            if self.log_spectrum.shape != self.slice_shape:
                raise ValueError(
                    f"Spectrum shape {self.log_spectrum.shape} does not match slice shape {self.slice_shape}"
                )

            self._adapt_filters_to_shape(height, width, previous_shape=previous_shape)
            self._resize_figure_for_image(height, width)
            axis_label = AXIS_LABELS.get(axis, str(axis))
            max_slice = info["shape"][axis] - 1
            self.status_var.set(
                f"Loaded {pathlib.Path(self.mhd_path).name} | volume={info['shape']} | "
                f"axis={axis_label} | slice={slice_index}/{max_slice} | "
                f"slice_size={width}x{height} | spectrum={width}x{height}"
            )
            self._draw_original()
            self._draw_spectrum()
            self.load_params_to_form()
            self.refresh_filter_list()
            self.update_markers()
            self.preview()
        except Exception as error:
            messagebox.showerror("Failed to load slice", str(error))

    def _draw_original(self):
        axis = self._current_axis()
        axis_label = AXIS_LABELS.get(axis, str(axis))
        self._display_image(
            self.ax_original,
            normalize_to_uint8(self.slice_image),
            f"Original Slice [{axis_label}]",
        )

    def _draw_spectrum(self):
        self._display_image(
            self.ax_spectrum,
            self.log_spectrum,
            "Frequency Spectrum (click/drag point)",
        )

    def _draw_filtered(self, filtered_image):
        if filtered_image.shape != self.slice_shape:
            raise ValueError(
                f"Filtered slice shape {filtered_image.shape} does not match original slice shape {self.slice_shape}"
            )
        self._display_image(
            self.ax_filtered,
            normalize_to_uint8(filtered_image),
            "Filtered Slice",
        )

    def update_markers(self):
        if self.log_spectrum is None:
            self._clear_markers()
            return

        self._clear_markers()

        spec = self.selected_filter()
        if spec is None:
            self.canvas.draw_idle()
            return

        height, width = self.log_spectrum.shape
        center_x = width / 2
        center_y = height / 2
        sym_x = 2 * center_x - spec.x
        sym_y = 2 * center_y - spec.y

        self.marker_artist = self.ax_spectrum.plot(
            spec.x,
            spec.y,
            marker="o",
            markersize=8,
            markerfacecolor="red",
            markeredgecolor="white",
            linestyle="None",
        )[0]
        self.symmetric_marker = self.ax_spectrum.plot(
            sym_x,
            sym_y,
            marker="x",
            markersize=8,
            color="yellow",
            linestyle="None",
        )[0]
        self.canvas.draw_idle()

    def preview(self):
        if self.fshift_base is None:
            return
        try:
            self.apply_parameters_silent()
            filtered, _ = apply_filter_specs(self.fshift_base, self.filters)
            if filtered.shape != self.slice_shape:
                raise ValueError(
                    f"Filtered shape {filtered.shape} does not match slice shape {self.slice_shape}"
                )
            self._draw_filtered(filtered)
            self._last_filtered = filtered
            self.update_markers()
            self.canvas.draw_idle()
            self._update_zoom_status()
            if self._large_view_window is not None and self._large_view_window.winfo_exists():
                self._refresh_large_filtered_view()
            active_count = sum(1 for item in self.filters if item.enabled)
            base = self.status_var.get().split(" | zoom=")[0]
            zoom = self._axis_zoom.get(id(self.ax_filtered), 1.0)
            self.status_var.set(f"{base} | active_filters={active_count} | zoom={zoom:.1f}x")
        except Exception as error:
            messagebox.showerror("Preview failed", str(error))

    def open_large_filtered_view(self):
        if self.fshift_base is None:
            messagebox.showinfo("Large View", "请先加载 MHD 并点击 Preview。")
            return
        if self._last_filtered is None:
            self.preview()
        if self._last_filtered is None:
            return

        if self._large_view_window is not None and self._large_view_window.winfo_exists():
            self._large_view_window.lift()
            self._refresh_large_filtered_view()
            return

        window = tk.Toplevel(self.root)
        window.title("Filtered Image - Large View")
        window.geometry("900x900")
        self._large_view_window = window

        fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = tk.Frame(window)
        toolbar.pack(fill=tk.X)
        tk.Label(
            toolbar,
            text="Ctrl + 滚轮放大/缩小 | 可查看微小噪声细节",
        ).pack(side=tk.LEFT, padx=8, pady=4)

        self._large_view_fig = fig
        self._large_view_ax = ax
        self._large_view_canvas = canvas
        self._large_view_zoom = 1.0
        self._large_view_limits = None

        def on_close():
            self._large_view_window = None
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", on_close)
        self._refresh_large_filtered_view()

        def on_large_scroll(event):
            direction = self._scroll_direction(event)
            if not self._ctrl_is_pressed(event) or direction == 0:
                return
            if self._large_view_limits is None:
                return
            zoom_in = direction > 0
            if zoom_in:
                self._large_view_zoom = min(self._max_zoom, self._large_view_zoom * 1.2)
            else:
                self._large_view_zoom = max(1.0, self._large_view_zoom / 1.2)
            bx0, bx1 = self._large_view_limits[0]
            by0, by1 = self._large_view_limits[1]
            cx = (bx0 + bx1) / 2
            cy = (by0 + by1) / 2
            half_w = (bx1 - bx0) / (2 * self._large_view_zoom)
            half_h = abs(by1 - by0) / (2 * self._large_view_zoom)
            ax.set_xlim(cx - half_w, cx + half_w)
            ax.set_ylim(cy + half_h, cy - half_h)
            canvas.draw_idle()

        canvas.mpl_connect("scroll_event", on_large_scroll)
        canvas.get_tk_widget().bind("<Control-MouseWheel>", on_large_scroll, add="+")
        canvas.get_tk_widget().bind("<Control-Button-4>", on_large_scroll, add="+")
        canvas.get_tk_widget().bind("<Control-Button-5>", on_large_scroll, add="+")

    def _refresh_large_filtered_view(self):
        if self._last_filtered is None or self._large_view_window is None:
            return
        if not self._large_view_window.winfo_exists():
            self._large_view_window = None
            return

        image = normalize_to_uint8(self._last_filtered)
        height, width = image.shape
        ax = self._large_view_ax
        ax.clear()
        ax.imshow(image, cmap="gray", aspect="equal", interpolation="nearest", origin="upper")
        xlim = (-0.5, width - 0.5)
        ylim = (height - 0.5, -0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"Filtered Slice - Large View ({width} x {height})")
        ax.set_xticks([])
        ax.set_yticks([])
        self._large_view_limits = (xlim, ylim)
        self._large_view_zoom = 1.0
        self._large_view_canvas.draw_idle()

    def apply_parameters_silent(self):
        spec = self.selected_filter()
        if spec is None:
            return
        try:
            spec.name = self.name_var.get().strip() or spec.name
            spec.filter_type = int(self.type_var.get())
            spec.x = float(self.x_var.get())
            spec.y = float(self.y_var.get())
            spec.radius = max(float(self.radius_var.get()), 1.0)
            spec.order = max(int(float(self.order_var.get())), 1)
            spec.enabled = self.enabled_var.get()
        except ValueError:
            return

    def on_canvas_press(self, event):
        if event.inaxes != self.ax_spectrum or self.selected_filter() is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.dragging = True
        self._set_selected_point(event.xdata, event.ydata, preview=True)

    def on_canvas_motion(self, event):
        if not self.dragging:
            return
        if event.inaxes != self.ax_spectrum:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._set_selected_point(event.xdata, event.ydata, preview=True)

    def on_canvas_release(self, _event):
        if self.dragging:
            self.dragging = False
            self.refresh_filter_list()

    def _set_selected_point(self, x_value, y_value, preview=False):
        spec = self.selected_filter()
        if spec is None:
            return
        spec.x = float(x_value)
        spec.y = float(y_value)
        self.x_var.set(f"{spec.x:.2f}")
        self.y_var.set(f"{spec.y:.2f}")
        self.update_markers()
        if preview:
            self.preview()

    def save_filtered(self):
        if self.fshift_base is None:
            messagebox.showinfo("Save", "Load an MHD slice and preview first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save filtered image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg")],
        )
        if not path:
            return
        filtered, _ = apply_filter_specs(self.fshift_base, self.filters)
        if filtered.shape != self.slice_shape:
            raise ValueError(
                f"Filtered shape {filtered.shape} does not match slice shape {self.slice_shape}"
            )
        save_slice_png(filtered, path)
        messagebox.showinfo("Saved", f"Filtered image saved to:\n{path}")

    def run(self):
        self.root.mainloop()


def main():
    pathlib.Path("tmp").mkdir(exist_ok=True)
    FilterPortalApp().run()


if __name__ == "__main__":
    main()
