#+
# :Author: Dr. Wei Huang (Email: huangwei@mail.bnu.edu.cn)
#-
import numpy as np
import os
from glob import glob
from netCDF4 import Dataset
from osgeo import osr
import gdal

NClist = glob(r'D:\xx\*.nc')
OutPath = r'D:\xx'

a = Dataset("CHAP_PM2.5_D1K_20170101_V4.nc")
print(a.variables.keys())
print("home")
for i in NClist:
    print("a")
    nc_obj = Dataset(i)
    fname = os.path.basename(i).split('.nc')[0]
    dictk = [v for v in nc_obj.variables.keys()]
    arr = nc_obj.variables[dictk[2]][:]
    result = np.zeros((arr.shape))
    c = arr.shape[1]
    for i in range(c):
        result[:,i] = np.array(list(arr[:,i].data)[::-1])
    lat= nc_obj.variables['lat'][:]
    lon= nc_obj.variables['lon'][:]
    LonMin,LatMax,LonMax,LatMin = lon.min(),lat.max(),lon.max(),lat.min()
    N_Lat = len(lat)
    N_Lon = len(lon)
    Lon_Res = round((LonMax-LonMin)/(float(N_Lon)-1),2)
    Lat_Res = round((LatMax-LatMin)/(float(N_Lat)-1),2)
    OutTif = OutPath + '/{}.tif' .format(fname)
    driver = gdal.GetDriverByName('Gtiff')
    outRaster = driver.Create(OutTif,N_Lon,N_Lat,1,gdal.GDT_Float32)
    geotransform = (LonMin,Lon_Res,0,LatMin-Lat_Res,0,Lat_Res)
    outRaster.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    outRaster.SetProjection(srs.ExportToWkt())
    outRaster.GetRasterBand(1).WriteArray(result)
    outRaster.FlushCache()
    outRaster = None
    print(fname+'.tif',' Finished')