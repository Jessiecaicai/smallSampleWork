# -*- coding: utf-8 -*-
# @Author  : Jessie
# @Time    : 2022/2/12 4:12 下午
# @Function:
import netCDF4
from netCDF4 import Dataset
# a = Dataset("/Desktop/code/yj/CHAP_PM2.5_D1K_20170101_V4.nc")
a = Dataset("CHAP_PM2.5_D1K_20170101_V4.nc")
print(a.variables.keys())