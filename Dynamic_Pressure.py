def Dynamic_Pressure(filename):

  # Import packages
  import xwrf
  import numpy as np
  import matplotlib.pyplot as plt
  from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                   cartopy_ylim, latlon_coords, interplevel)
  import xarray as xr
  from netCDF4 import Dataset

  from matplotlib.cm import get_cmap
  import cartopy.crs as crs
  import cartopy.feature as cfeature
  from metpy.plots import colortables, USSTATES, USCOUNTIES

  # Open the NetCDF file
  ncfile = Dataset(filename)

  # Get neccessary variables
  h = getvar(ncfile, 'height_agl')
  wa = getvar(ncfile, "wa", units="m s-1")
  ua = getvar(ncfile, "ua", units="m s-1")
  va = getvar(ncfile, "va", units="m s-1")

  # Get shape of array and set vortex location
  x_len = h.shape[2]
  y_len = h.shape[1]
  center_x = 158
  center_y = 184

  # Make list of heights
  hgt_list = np.arange(500, 18000, 250)
  add = [80, 100, 200, 300, 400]
  hgt_list = np.insert(hgt_list, 0, add)

  # Generate empty arrays to save data
  OW_array = np.zeros((len(hgt_list), y_len, x_len))
  u_array = np.zeros((len(hgt_list), y_len, x_len))
  v_array = np.zeros((len(hgt_list), y_len, x_len))
  w_array = np.zeros((len(hgt_list), y_len, x_len))
  linear_force_array = np.zeros((len(hgt_list), y_len, x_len))
  nonlinear_force_array = np.zeros((len(hgt_list), y_len, x_len))

  # Loop through heights to obtain values at each level
  for i, level in enumerate(hgt_list):
      # Interpolate wind velocities to level
      u_level = to_np(interplevel(ua, h, level))
      v_level = to_np(interplevel(va, h, level))
      w_level = to_np(interplevel(wa, h, level))

      # Save these arrays
      u_array[i,:,:]=u_level
      v_array[i,:,:]=v_level
      w_array[i,:,:]=w_level

      # Calculate OW Parameter
      dx = 1000 # Meters, calculated from 1km grid spacing
      dy = 1000 # Meters, calculated from 1km grid spacing
      du_dy, du_dx = np.gradient(u_level, dy,dx)
      dv_dy, dv_dx = np.gradient(v_level, dy,dx)
      D1 = du_dx-dv_dy
      D2 = dv_dx+du_dy
      Vort = dv_dx-du_dy
      D = np.sqrt(np.square(D1)+np.square(D2))
      OW = (np.square(D)-np.square(Vort))
      OW = OW*-10000 # Plot as 10^-4 s^-2 and reversed
      OW_array[i,:,:]=OW

      # Obtain wind velocities above and below the given level
      level_top = level+50
      level_bottom = level-50
      u_top = to_np(interplevel(ua, h, level_top))
      u_bottom = to_np(interplevel(ua, h, level_bottom))
      v_top = to_np(interplevel(va, h, level_top))
      v_bottom = to_np(interplevel(va, h, level_bottom))
      w_top = to_np(interplevel(wa, h, level_top))
      w_bottom = to_np(interplevel(wa, h, level_bottom))

      # Obtain a base state set of values
      base_center_x = center_x+30
      base_center_y = center_y-30
      u_base_level = u_level[base_center_y-10:base_center_y+10, base_center_x-10:base_center_x+10]
      u_base_top = u_top[base_center_y-10:base_center_y+10, base_center_x-10:base_center_x+10]
      u_base_bottom = u_bottom[base_center_y-10:base_center_y+10, base_center_x-10:base_center_x+10]
      v_base_level = v_level[base_center_y-10:base_center_y+10, base_center_x-10:base_center_x+10]
      v_base_top = v_top[base_center_y-10:base_center_y+10, base_center_x-10:base_center_x+10]
      v_base_bottom = v_bottom[base_center_y-10:base_center_y+10, base_center_x-10:base_center_x+10]

      # Calculate linear dynamics pressure at this level
      dw_dy, dw_dx = np.gradient(w_level, dy, dx)
      dU = u_base_top.mean() - u_base_bottom.mean()
      #dU = dU.mean()
      dV = v_base_top.mean() - v_base_bottom.mean()
      #dV = dV.mean()
      dz = level_top - level_bottom
      linear_dynamics = -1*(((dU/dz)*dw_dx)+((dV/dz)*dw_dy))
      linear_force_array[i,:,:]=linear_dynamics

      # Calculate nonlinear dynamics pressure at this level
      du_dy, du_dx = np.gradient(u_level, dy, dx)
      dv_dy, dv_dx = np.gradient(v_level, dy, dx)
      dw = w_top - w_bottom
      du = u_top - u_bottom
      dv = v_top - v_bottom
      dz = level_top - level_bottom
      nonlinear_dynamics = -1*((du_dx**2)+(dv_dy**2)+((dw/dz)**2)+
                               (2*((du_dy*dv_dx)+((du/dz)*dw_dx)+((dv/dz)*dw_dy))))
      nonlinear_force_array[i,:,:]=nonlinear_dynamics
      
      return u_array, v_array, w_array, OW_array, linear_force_array, nonlinear_force_array
