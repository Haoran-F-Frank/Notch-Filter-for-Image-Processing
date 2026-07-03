# Notch-Filter-for-Image-Processing
Three varients of the Notch Filters are implemented:
  * Ideal Notch Filter
  * ButterWorth Notch Filter (Different orders are also supported)
  * Gaussian Notch Filter
# How to run 
  1. Clone this repo. 
  2. Fulfill ```requirements.txt``` (```pip install -r requirements.txt```).
  3. Run GUI with command ```python3 main.py```.
  4. Run interactive MHD portal with ```python3 portal.py```.

# MHD Interactive Filter Portal

`portal.py` is an interactive tool for `.mhd` / `.mha` / `.raw` biomedical images:

- Load an MHD volume and choose a slice
- Switch slice direction with **Axis** (0=Z/axial, 1=Y/coronal, 2=X/sagittal)
- Create, move, delete, and duplicate notch filters
- Filter types: `0 = Butterworth`, `1 = Gaussian`
- Butterworth order is adjustable per filter
- Click or drag on the frequency spectrum to position each filter point
- Mouse wheel scrolls through slices; Ctrl + mouse wheel zooms the image view
- Adjust display intensity window (default -500 to 1000) without changing FFT/filter data
- Preview filtered result side-by-side with original image and spectrum
- Save filtered slice as PNG or full filtered volume as MHD (raw intensity preserved)
- Original and filtered time-domain images always share the same display intensity window
- Resize original/spectrum/filtered panels with width sliders
- Click original/filtered to zoom in at cursor; Shift+click zooms out
- Display keeps the original slice aspect ratio (non-square images are not stretched)
- Frequency spectrum and filtered output always keep the same width x height as the input slice

```bash
pip install -r requirements.txt
python3 portal.py
```

# Demo Run

1. Running ```main.py``` will give follwing screen:

<p align="center">
  <img src="https://github.com/imdeep2905/Notch-Filter-for-Image-Processing/blob/master/imgs/Demo1.PNG">
</p>

2. Select an image on which you want to apply the filter.

<p align="center">
  <img src="https://github.com/imdeep2905/Notch-Filter-for-Image-Processing/blob/master/imgs/Demo2.PNG">
</p>

3. Select points to apply filter.

<p align="center">
  <img src="https://github.com/imdeep2905/Notch-Filter-for-Image-Processing/blob/master/imgs/Demo3.PNG">
</p>

<p align="center">
  <img src="https://github.com/imdeep2905/Notch-Filter-for-Image-Processing/blob/master/imgs/Demo4.PNG">
</p>

4. Side by side comparison.

<p align="center">
  <img src="https://github.com/imdeep2905/Notch-Filter-for-Image-Processing/blob/master/imgs/Demo5.PNG">
</p>

<p align="center">
  <img src="https://github.com/imdeep2905/Notch-Filter-for-Image-Processing/blob/master/imgs/Demo6.PNG">
</p>

# Some Examples

<p align="center">
  <img src="https://github.com/imdeep2905/Notch-Filter-for-Image-Processing/blob/master/imgs/Working1.PNG">
</p>

<p align="center">
  <img src="https://github.com/imdeep2905/Notch-Filter-for-Image-Processing/blob/master/imgs/Working2.PNG">
</p>

<p align="center">
  <img src="https://github.com/imdeep2905/Notch-Filter-for-Image-Processing/blob/master/imgs/Working3.PNG">
</p>

<p align="center">
  <img src="https://github.com/imdeep2905/Notch-Filter-for-Image-Processing/blob/master/imgs/Working4.PNG">
</p>


