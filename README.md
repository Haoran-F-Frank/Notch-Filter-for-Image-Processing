# Notch-Filter-for-Image-Processing
Three varients of the Notch Filters are implemented:
  * Ideal Notch Filter
  * ButterWorth Notch Filter (Different orders are also supported)
  * Gaussian Notch Filter

# How to run 
  1. Clone this repo. 
  2. Fulfill ```requirements.txt``` (```pip install -r requirements.txt```).
  3. Run GUI with command ```python3 main.py```.
  4. On Linux, install Tkinter if needed: ```sudo apt install python3-tk```.

# Biomedical image support (NIfTI / MetaImage)

Supported formats:
  * `.nii`, `.nii.gz` (NIfTI)
  * `.mhd`, `.mha`, `.raw` (MetaImage; `.raw` requires a matching `.mhd` header in the same folder)

For 3D volumes, choose a slice index in the GUI or pass `--slice` in the CLI.

# Terminal / CLI usage

Analyze noise in the frequency domain:

```bash
python3 cli.py freq your_image.nii.gz --slice 64 --output tmp/dft.png
```

Apply notch filtering from the terminal:

```bash
python3 cli.py denoise your_image.mhd --slice 32 --auto-peaks 6 --filter butterworth --radius 121 --output tmp/filtered.png
```

Manual notch points (x,y pairs, same coordinate system as the GUI click points):

```bash
python3 cli.py denoise scan.nii.gz --points "120,80;300,200" --filter gaussian --radius 100
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


