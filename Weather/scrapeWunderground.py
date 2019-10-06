# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 06:58:41 2019

@author: tom_m
"""

locations = {
        'darwin': 'https://www.wunderground.com/forecast/au/darwin/-12.46,130.84'
        }

from bs4 import BeautifulSoup
soup = BeautifulSoup(html_doc, 'html.parser')

