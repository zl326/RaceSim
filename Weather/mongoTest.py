# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:18:02 2019

@author: tom_m
"""

import pymongo

client = pymongo.MongoClient('localhost', 27017)

db = client['CUER']

collection = db['weather']

a = collection.find({
    "_docType": "hourly",
    "time": {
        "$gte": 1571134400,
        "$lt": 1571136400
    }
})

# 1571135400

print(a)

for i in a :
    print(i)
    print('')


