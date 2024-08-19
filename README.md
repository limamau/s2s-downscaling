# S2S Downscaling
This repository contains ECCE's downscaling code for Subseasonal-to-seasonal (S2S) forecasts.

### Installation
Create a virtual environment with all the dependencies:
```
pip install -r requirements.txt
```

Then build the project:
```
pip install -e .
```

### Examples
We provide a simple example on how to use a diffusion model adapted from [swirl-dynamics](https://github.com/google-research/swirl-dynamics/tree/main/swirl_dynamics/projects/probabilistic_diffusion).

The main example uses [CombiPrecip](https://www.meteoswiss.admin.ch/services-and-publications/service/weather-and-climate-products/combiprecip.html) for the high resolution and [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form) for the low resolution datasets. Additional configurations for a [WRF](https://www.mmm.ucar.edu/models/wrf) run and comparison are included.