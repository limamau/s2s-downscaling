# S2S Downscaling
This repository contains ECCE's downscaling code for Subseasonal-to-seasonal (S2S) forecasts.

### Installation
Create a conda environment with all the dependencies. We assume cuda/12.2.1 is installed on your computer.
```
conda env create -f environment.yml
```

Then build the project.
```
pip install -e .
```

### Examples 
We currently provide one example with [CombiPrecip](https://www.meteoswiss.admin.ch/services-and-publications/service/weather-and-climate-products/combiprecip.html) for the high resolution and [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form) for the low resolution datasets. Additional configurations for a [WRF](https://www.mmm.ucar.edu/models/wrf) run and comparison are included.