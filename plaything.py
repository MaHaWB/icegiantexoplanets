# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 20:19:19 2021

@author: marcu
"""
from astropy import units
import astropy.coordinates as coord

# JPL Horizons: 2040-Dec-01, Neptune-Sun, icrf
ra= '+13d29m43.3s'
dec= '+47d11m34.7s'

c = coord.SkyCoord(ra=ra,
                   dec=dec,
                   frame='icrs')

print(c.transform_to(coord.BarycentricMeanEcliptic).lat.radian)