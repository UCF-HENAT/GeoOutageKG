from gadm import GADMDownloader
from blackmarble.download import BlackMarbleDownloader
import pickle
from pathlib import Path
import geopandas as gpd
import os
import pandas as pd
import xarray as xr
from shapely.vectorized import contains
import numpy as np

bearer = os.getenv('EARTH_DATA_TOKEN')
usa_gdf = GADMDownloader(version="4.0").get_shape_data_by_country_name(country_name="USA", ad_level=2)
florida_gdf = usa_gdf[usa_gdf["NAME_1"] == 'Florida']
H5_CACHE = os.path.expanduser("./h5_cache/")
os.makedirs(H5_CACHE, exist_ok=True)

def download_all_counties(product="VNP46A3", date_range=None):
  county_names = list(florida_gdf["NAME_2"].values)
  county_names = [name.lower().replace(' ', '_').replace('-', '_') for name in county_names]

  freq = 'MS' if product == "VNP46A3" else 'D'
  date_range_pd = pd.date_range(start=date_range[0], end=date_range[1], freq=freq)

  # download manager
  bm = BlackMarbleDownloader(bearer=bearer, directory=H5_CACHE)

  # perform downloads
  h5_files = bm.download(
      gdf=florida_gdf,
      product_id=product,        # daily composites
      date_range=date_range_pd,
      skip_if_exists=False
  )
  
  for county_name in county_names:
    try:
        # select this county's geometry
      county_gdf = florida_gdf[florida_gdf['NAME_2']
          .str.lower().str.replace(' ', '_').str.replace('-', '_') == county_name]
      poly = county_gdf.geometry.union_all()

      if product == "VNP46A3":
        download_county_data_vnp46a3(h5_files, county_name, poly)
      elif product == "VNP46A2":
        download_county_data_vnp46a2(h5_files, county_name, poly)
      else:
        print("Invalid product given.")
        return
    except Exception as e:
      print(f"Failed on county {county_name}: {e}")

def download_county_data_vnp46a2(h5_files, county_name, poly):
  """
  Download daily VNP46A2 data (Gap_Filled_DNB_BRDF-Corrected_NTL) for one Florida county.

  Args:
      county_name (str): normalized county name (lower, underscores)
      date_range (pd.DatetimeIndex): daily dates to download
  """

  DATA_PATH = "./county_VNP46A2"
  # prepare output directory
  county_path = os.path.join(DATA_PATH, county_name)
  os.makedirs(county_path, exist_ok=True)

  # read & mask each day
  for h5 in h5_files:
    try:
      print(h5)
      ds = xr.open_dataset(
          h5,
          engine="h5netcdf",
          group="HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/",
          backend_kwargs={"phony_dims": "sort"}
      )

      coord_ds = xr.open_dataset(
          f"./h5_cache/VNP46A3.A2024336.{str(h5)[-27:-21]}.h5",
          engine="h5netcdf",
          group="HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/",
          backend_kwargs={"phony_dims": "sort"}
      )
      lat = coord_ds['lat'].values
      lon = coord_ds['lon'].values
      lat1d = lat[::-1]
      lon1d = lon   # east-west doesn't need flipping

      doy = h5.stem.split('.')[1]
      time = pd.to_datetime(doy, format="A%Y%j")
      da = ds["Gap_Filled_DNB_BRDF-Corrected_NTL"]
      da = da.rename({
          da.dims[0]: "y",
          da.dims[1]: "x"
      })
      da = da.isel(y=slice(None, None, -1))
      da = da.assign_coords(y=("y", lat1d), x=("x", lon1d))\
              .rename({"y": "latitude", "x": "longitude"})
      
      lon2d, lat2d = np.meshgrid(lon1d, lat1d)
      mask = contains(poly, lon2d, lat2d)
      if not mask.any():
          print(f"  → no overlap for {county_name} on {time.strftime('%Y%m%d')}, skipping")
          continue

      mask_da = xr.DataArray(mask, dims=("latitude","longitude"),
                              coords={"latitude": da.latitude, "longitude": da.longitude})
      da = da.where(mask_da, drop=True)
      da = da.expand_dims(time=[time])

      data_bytes = pickle.dumps(da)
      new_size = len(data_bytes)

      # save each day separately
      out_fp = os.path.join(county_path, f"{time.strftime('%Y_%m_%d')}.pickle")
      if os.path.exists(out_fp):
        old_size = os.path.getsize(out_fp)
        if new_size > old_size:
          with open(out_fp, 'wb') as f:
              pickle.dump(da, f)
      else:
        with open(out_fp, 'wb') as f:
            pickle.dump(da, f)

    except Exception as e:
      print(f"Failed on file {h5}: {e}")

def download_county_data_vnp46a3(h5_files, county_name, poly):
    """
    Download monthly VNP46A3 data for one Florida county,
    mosaicking all tiles per month into one array before masking.
    """
    DATA_PATH = "./county_VNP46A3"
    county_path = os.path.join(DATA_PATH, county_name)
    os.makedirs(county_path, exist_ok=True)
    

    # 1) Group all files by their month
    files_by_month = {}
    for h5 in h5_files:
        stem = Path(h5).stem                   # e.g. "VNP46A3.A2022152.h09v05"
        doy   = stem.split(".")[1]             # "A2022152"
        dt    = pd.to_datetime(doy, format="A%Y%j")
        month = dt.to_period("M")
        files_by_month.setdefault(month, []).append((h5, dt))

    monthly_arrays = []
    for month, month_files in sorted(files_by_month.items()):
        tile_datasets = []
        for h5, dt in month_files:
            ds = xr.open_dataset(
                h5,
                engine="h5netcdf",
                group="HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/",
                backend_kwargs={"phony_dims": "sort"}
            )

            da = ds["NearNadir_Composite_Snow_Free"] \
                   .rename({"phony_dim_0": "y", "phony_dim_1": "x"}) \
                   .isel(y=slice(None, None, -1))

            # rebuild 1-d coords
            lat1d = ds["lat"].values[::-1]
            lon1d = ds["lon"].values
            da = da.assign_coords(y=("y", lat1d), x=("x", lon1d)) \
                   .rename({"y": "latitude", "x": "longitude"})

            # wrap into a Dataset so we can merge tiles
            tile_datasets.append(da.to_dataset())

        if not tile_datasets:
            continue

        # 2) Mosaic all tiles into one Dataset (outer join on lat/lon)
        mosaic_ds = xr.merge(tile_datasets, join="outer")

        # 3) Extract the combined DataArray
        mosaic_da = mosaic_ds["NearNadir_Composite_Snow_Free"]

        # 4) Put back the proper time coordinate (first of month)
        time = month.to_timestamp()  # e.g. Timestamp('2022-06-01 00:00:00')
        mosaic_da = mosaic_da.expand_dims(time=[time])

        # 5) Mask to the county polygon
        lon2d, lat2d = np.meshgrid(mosaic_da.longitude.values,
                                   mosaic_da.latitude.values)
        mask = contains(poly, lon2d, lat2d)
        mask_da = xr.DataArray(mask,
                               dims=("latitude", "longitude"),
                               coords={
                                   "latitude": mosaic_da.latitude,
                                   "longitude": mosaic_da.longitude
                               })
        mosaic_da = mosaic_da.where(mask_da, drop=True)

        monthly_arrays.append(mosaic_da)

    if monthly_arrays:
        # 6) Concatenate the monthly mosaics along time
        vnp46a3_data = xr.concat(monthly_arrays, dim="time")

        # 7) Save one pickle per county per year (or adjust filename as you like)
        year = monthly_arrays[0].time.dt.year.values.tolist()[0]
        out_fp = os.path.join(county_path, f"{county_name}_{year}.pickle")
        with open(out_fp, "wb") as f:
            pickle.dump(vnp46a3_data, f)
        print(f"  → saved {out_fp}")
    else:
        print(f"  → no data for {county_name}")

if __name__ == "__main__":
  download_all_counties(product="VNP46A3", date_range=("2024-06-01", "2024-12-31"))