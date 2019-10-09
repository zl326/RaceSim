import pandas as pd
import yaml
import datetime
from astral import Location
import dateutil
import math
import numpy as np
from scipy.interpolate import griddata
import os
import copy
import pymongo

class Simulation:
    
    kph2ms = 1/3.6
    ms2kph = 3.6
    rad2deg = 180.0/math.pi
    deg2rad = math.pi/180.0
    g = 9.80665
    Ra = 286.9
    Rw = 461.5
    C2K = 273.15
    K2C = -273.15
    rads2RPM = 60/(2*math.pi)
    RPM2rads = 2*math.pi/60
    
    def __init__(self, settingsPath):

        # Load settings
        self.settingsPath = settingsPath
        with open(settingsPath, 'r') as stream:
            try:
                self.settings = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        # Initialise MongoDB connection
        if self.settings['weather']['fromMongo'] :
            self.mongClient = pymongo.MongoClient('localhost', 27017)
            db = self.mongClient['CUER']
            self.db_weather = db['weather']
        
        self.initData()
        
    def initData(self):
        self.getRoute()
        self.getWeatherData()
        self.getSolarProjectedAreaData()
        self.getElectricalData()
        
        self.updateCol(self.data, 'simName', self.settings['meta']['name'])
        
        # Add meta data to data table
        
        self.splitStints()
    
    def getRoute(self):
        self.data = pd.read_csv(self.settings['route']['routeFile'])
        
        # Set the speed constraints along the route
        self.updateCol(self.data, 'speedMin', 0)
        self.updateCol(self.data, 'speedMax', 999)
        
        iConstraint = 0
        
        # Loop through all mesh points
        for i in range(0, len(self.data)):
            
            # Calculate heading using central difference
            i1 = i
            i2 = i
            if i == 0:
                i1 = i
                i2 = i+1
            elif i == (len(self.data)-1):
                i1 = i-1
                i2 = i
            else:
                 i1 = i-1
                 i2 = i+1
            
            d_long = self.data['longitude'][i2] - self.data['longitude'][i1]
            d_lat = self.data['latitude'][i2] - self.data['latitude'][i1]
            heading = math.atan2(d_long, d_lat)*self.rad2deg
            heading = heading + (heading<0)*360
            
            self.data.at[i, 'heading'] = heading
            
            # Apply speed constraints
            if iConstraint >= len(self.settings['route']['speedConstraints']):
                # No more constraints
                break
            else:
                if self.data.distance[i] >= self.settings['route']['speedConstraints'][iConstraint]['startDistance']:
                    if iConstraint < len(self.settings['route']['speedConstraints'])-1:
                        # This is not the last constraint
                        if self.data.distance[i] >= self.settings['route']['speedConstraints'][iConstraint+1]['startDistance']:
                            iConstraint += 1
                    
                    # At this point we have the correct constraint identified
                    self.data.at[i, 'speedMin'] = self.settings['route']['speedConstraints'][iConstraint]['speedMin']
                    self.data.at[i, 'speedMax'] = self.settings['route']['speedConstraints'][iConstraint]['speedMax']
            
        # Convert the locations list to a more user friendly dict
        locations = self.settings['route']['locations']
        self.locations = {}
        for iLocation in range(0, len(locations)):
            thisLocation = locations[iLocation]
            self.locations[thisLocation['name']] = {}
            self.locations[thisLocation['name']]['distance'] = thisLocation['distance']
    
    def getSolarProjectedAreaData(self):
        
        self.solarprojectedAreaX = pd.read_csv(self.settings['solar']['projectedAreaXFilePath'])
        self.solarprojectedAreaY = pd.read_csv(self.settings['solar']['projectedAreaYFilePath'])
        
        
        yRotationPattern = self.solarprojectedAreaY.rotation.to_numpy()
        
        yRotation = np.array([])
        xRotation = np.array([])
        areaRatio = np.array([])
        
        # Compile the full grid
        for iXRotationValue in range(0, len(self.solarprojectedAreaX.rotation)):
        
            xRotationValue = self.solarprojectedAreaX.rotation[iXRotationValue]
            
            yRotation = np.concatenate((yRotation, yRotationPattern))
            xRotation = np.concatenate((xRotation, yRotationPattern*0+xRotationValue))
            
            areaRatioAtThisXRotation = self.solarprojectedAreaY.areaRatio.to_numpy() * self.solarprojectedAreaX.areaRatio[iXRotationValue]
            areaRatio = np.concatenate((areaRatio, areaRatioAtThisXRotation))
            
        self.solarXRotation = xRotation
        self.solarYRotation = yRotation
        self.solarAreaRatio = areaRatio
        
    def getElectricalData(self):
        self.efficiencyMotorController = pd.read_csv(self.settings['powertrain']['waveSculptorEfficiencyFilePath'])
        self.batteryCellDischargeCurve = pd.read_csv(self.settings['battery']['cellDischargeCurveFilePath'])
        
        # Initialise battery SOC
        self.updateCol(self.data, 'car__rSOC', 1)
    
    def getWeatherData(self):
        
        print('Loading weather data...'.format())
        
        if self.settings['weather']['fromCsv']:
            self.weatherData = pd.read_csv(self.settings['weather']['filePath'])
            
            for i in range(0, len(self.weatherData)):
                # Convert wind direction cardinals to degrees
                if isinstance(self.weatherData.windDirection[0], str):
                    self.weatherData.at[i, 'windDirectionDeg'] = self.settings['compass']['cardinal2deg'][self.weatherData.windDirection[i]]
                    
                    # Convert wind speed into northerly and easterly components
                    self.weatherData.at[i, 'windCompN'] = math.cos(self.weatherData['windDirectionDeg'][i]*self.deg2rad)
                    self.weatherData.at[i, 'windCompE'] = math.sin(self.weatherData['windDirectionDeg'][i]*self.deg2rad)
                
                # Convert to datetime
                self.weatherData.at[i, 'datetime'] = dateutil.parser.parse(self.weatherData.time[i])
                
                # Assign a distance to each location
                self.weatherData.at[i, 'distance'] = self.locations[self.weatherData.location[i]]['distance']
                
        elif self.settings['weather']['fromMongo']:
            
            self.weatherCursor = self.db_weather.find({
                "_docType": "hourly",
                "time": {
                    "$gte": self.settings['time']['days'][0]['start'].timestamp() - 8.5*3600 - 1.0*3600,
                    "$lte": self.settings['time']['days'][len(self.settings['time']['days'])-1]['end'].timestamp() - 8.5*3600 + 1.0*3600
                }
            })
            
            # Prepare mongo data for interpolation
            distance = np.array([])
            time = np.array([])
            
            airTemp = np.array([])
            airPressure = np.array([])
            humidity = np.array([])
            windSpeed = np.array([])
            windGust = np.array([])
            windDirection = np.array([])
            windHeading = np.array([])
#            airDensity = np.array([])
            cloudCover = np.array([])
            precipProbability = np.array([])
            precipIntensity = np.array([])
            
            for doc in self.weatherCursor :
                distance = np.append(distance, doc['_distance'])
                time = np.append(time, doc['time'] + 9.5*3600)
                
                airTemp = np.append(airTemp, doc['temperature'])
                airPressure = np.append(airPressure, doc['pressure'] * 1E2)
                humidity = np.append(humidity, doc['humidity'])
                windSpeed = np.append(windSpeed, doc['windSpeed'] * self.ms2kph)
                windGust = np.append(windGust, doc['windGust'] * self.ms2kph)
                windDirection = np.append(windDirection, doc['windBearing'] - 180)
                windHeading = np.append(windHeading, doc['windBearing'])
#                airDensity = np.append(airDensity, doc[''])
                cloudCover = np.append(cloudCover, doc['cloudCover'])
                precipProbability = np.append(precipProbability, doc['precipProbability'])
                precipIntensity = np.append(precipIntensity, doc['precipIntensity'])
                
            self.weather = {}
                
            self.weather['d_min'] = np.min(distance)
            d_max = np.max(distance)
            self.weather['d_range'] = d_max - self.weather['d_min']
            self.weather['d_norm'] = (distance - self.weather['d_min']) / self.weather['d_range']
            
            self.weather['t_min'] = np.min(time)
            t_max = np.max(time)
            self.weather['t_range'] = t_max - self.weather['t_min']
            self.weather['t_norm'] = (time - self.weather['t_min']) / self.weather['t_range']
            
            self.weather['airTemp'] = airTemp
            self.weather['airPressure'] = airPressure
            self.weather['humidity'] = humidity
            self.weather['windSpeed'] = windSpeed
            self.weather['windGust'] = windGust
            self.weather['windDirection'] = windDirection
            self.weather['windHeading'] = windHeading
            self.weather['cloudCover'] = cloudCover
            self.weather['precipProbability'] = precipProbability
            self.weather['precipIntensity'] = precipIntensity
            
        print('Loading weather data... Complete'.format())
            
                
    def splitStints(self):
        # Set the stints
        self.stints = self.settings['route']['stints']
        self.NStints = len(self.stints)
        
        for iStint in range(0, self.NStints):
            stintNumber = iStint+1
            
            startDistance = -1
            endDistance = -1
                
            startDistance = max(self.settings['initialConditions']['distance'], self.locations[self.stints[iStint]['startLocation']]['distance'])
            endDistance = self.locations[self.stints[iStint]['endLocation']]['distance']
            
            # Meta data
            self.stints[iStint]['isSensitivities'] = False
            self.stints[iStint]['nStint'] = stintNumber
            self.stints[iStint]['startDistance'] = startDistance
            self.stints[iStint]['endDistance'] = endDistance
            
            # Find the section of the table that contains the current stint
            inStint = ((self.data.distance > startDistance) & (self.data.distance <= endDistance)).values.tolist()
            if True in inStint:
                startIndex = inStint.index(True) - 1
                endIndex = len(inStint) - inStint[::-1].index(True) - 1
            
                # Get the data that belongs to this stint
                self.stints[iStint]['data'] = self.data.iloc[startIndex:endIndex+1, :]
            
                # Reset the index
                self.stints[iStint]['data'] = self.stints[iStint]['data'].reset_index(drop=True)
                
                # Add some meta data
                self.stints[iStint]['stintLength'] = endDistance - startDistance
                self.stints[iStint]['meshPoints'] = endIndex+1-startIndex
                
                # Calculate additional route properties
                self.stints[iStint]['data'].at[0, 'd_distance'] = 0
                self.stints[iStint]['data'].at[0, 'd_elevation'] = 0
                self.stints[iStint]['data'].at[0, 'inclination_angle'] = 0
                self.stints[iStint]['data'].at[0, 'inclination_angle_deg'] = 0
                for i in range(1, len(self.stints[iStint]['data'])):
                    # Set the distance between mesh locationss
                    self.stints[iStint]['data'].at[i, 'd_distance'] = self.stints[iStint]['data'].loc[i, 'distance'] - self.stints[iStint]['data'].loc[i-1, 'distance']
                    
                    # Calculate change in elevation
                    self.stints[iStint]['data'].at[i, 'd_elevation'] = self.stints[iStint]['data'].loc[i, 'elevation'] - self.stints[iStint]['data'].loc[i-1, 'elevation']
                    self.stints[iStint]['data'].at[i, 'inclination_angle'] = math.atan(self.stints[iStint]['data'].at[i, 'd_elevation']/(self.stints[iStint]['data'].at[i, 'd_distance']*1e3))
                    self.stints[iStint]['data'].at[i, 'inclination_angle_deg'] = self.rad2deg*math.atan(self.stints[iStint]['data'].at[i, 'd_elevation']/(self.stints[iStint]['data'].at[i, 'd_distance']*1e3))
                
                # Set the stint number (starting from 1)
                self.stints[iStint]['data'].insert(len(self.stints[iStint]['data'].columns), 'stintNumber', stintNumber)
                
                # Define the stint start time
                if self.settings['initialConditions']['time'] > self.settings['time']['days'][self.stints[iStint]['startDayDefault']-1]['start']:
                    self.stints[iStint]['startTime'] = self.settings['initialConditions']['time']
                else:
                    self.stints[iStint]['startTime'] = self.settings['time']['days'][self.stints[iStint]['startDayDefault']-1]['start']
                
                self.stints[iStint]['startDay'] = self.stints[iStint]['startDayDefault']
                for iDay in range(0, len(self.settings['time']['days'])):
                    # Find the start day
                    if (self.stints[iStint]['startTime'] >= self.settings['time']['days'][iDay]['start']) & (self.stints[iStint]['startTime'] < self.settings['time']['days'][iDay]['end']) :
                        self.stints[iStint]['startDay'] = iDay+1
                    
                    # Find the end day
                    if (self.stints[iStint]['arrivalTime'] > self.settings['time']['days'][iDay]['start']) & (self.stints[iStint]['arrivalTime'] <= self.settings['time']['days'][iDay]['end']) :
                        self.stints[iStint]['endDay'] = iDay+1
                
                # Control stops
                self.stints[iStint]['NControlStops'] = len(self.stints[iStint]['controlStops'])
                
                # Calculate the time available
                # Time available = Total allowable time on road - time for control stops
                self.stints[iStint]['availableTime'] = datetime.timedelta(0) # Initialise time available
                for iDay in range(self.stints[iStint]['startDay']-1, self.stints[iStint]['endDay']):
                    if iDay == self.stints[iStint]['startDay']-1:
                        # First day of stint
                        if self.settings['time']['days'][iDay]['end'] > self.stints[iStint]['startTime']:
                            self.stints[iStint]['availableTime'] += self.settings['time']['days'][iDay]['end'] - self.stints[iStint]['startTime']
                        
                    elif iDay == self.stints[iStint]['endDay']-1:
                        # Last day of stint
                        if self.settings['time']['days'][iDay]['start'] > self.stints[iStint]['startTime']:
                            self.stints[iStint]['availableTime'] += self.stints[iStint]['arrivalTime'] - self.settings['time']['days'][iDay]['start']
                        else:
                            self.stints[iStint]['availableTime'] += self.stints[iStint]['arrivalTime'] - self.stints[iStint]['startTime']
                        
                    else:
                        # Day between start and end days (not applicable for 2019)
                        if self.settings['time']['days'][iDay]['start'] > self.stints[iStint]['startTime']:
                            self.stints[iStint]['availableTime'] += self.settings['time']['days'][iDay]['end'] - self.settings['time']['days'][iDay]['start']
                        else:
                            self.stints[iStint]['availableTime'] += self.settings['time']['days'][iDay]['end'] - self.stints[iStint]['startTime']
                
                self.stints[iStint]['availableTime'] -= self.stints[iStint]['NControlStops']*datetime.timedelta(minutes=self.settings['time']['controlStops']['duration'])
                
                
                # Initialise the speed
                self.stints[iStint]['averageSpeed'] = self.stints[iStint]['stintLength'] / (self.stints[iStint]['availableTime'].seconds/3600)
                self.updateCol(self.stints[iStint]['data'], 'speed', self.stints[iStint]['averageSpeed'])
            
    def runModels(self, stint):
        self.getWeather(stint)
        self.calculateAero(stint)
        self.calculateMech(stint)
        self.calculateElec(stint)
        self.calculateSolar(stint)
        self.calculateEnergy(stint)
    
    def calculateTime(self, stint):
        # Calculates the time using the car speed as input
        
        self.updateCol(stint['data'], 'time', stint['startTime'])
        self.updateCol(stint['data'], 'day', stint['startDay'])
        self.updateCol(stint['data'], 'time_unix', stint['startTime'].timestamp())
        self.updateCol(stint['data'], 'd_time', 0)
        self.updateCol(stint['data'], 'd_timeDriving', 0)
        
        self.updateCol(stint['data'], 'solar__sunElevationAngle', 0)
        self.updateCol(stint['data'], 'solar__sunAzimuthAngle', 0)
        
        for i in range(0, len(stint['data'])):
            
            if i > 0:
                averageSpeed = 0.5*stint['data'].speed[i] + 0.5*stint['data'].speed[i-1]
                
                d_time = datetime.timedelta(hours=stint['data'].d_distance[i]/averageSpeed)


                d_timeDriving = d_time
                
                # Account for controls stops
                for iControlStop in range(0,stint['NControlStops']):
                    if (stint['data'].distance[i-1] <= self.locations[stint['controlStops'][iControlStop]]['distance']) & (stint['data'].distance[i] > self.locations[stint['controlStops'][iControlStop]]['distance']):
                        d_time += datetime.timedelta(minutes=self.settings['time']['controlStops']['duration'])
                
                # Account for end of day
                if stint['data'].day[i] < len(self.settings['time']['days']):
                    if stint['data'].at[i-1, 'time'] + d_time > self.settings['time']['days'][stint['data'].day[i]-1]['end']:
                        d_time += self.settings['time']['days'][stint['data'].day[i]]['start'] - self.settings['time']['days'][stint['data'].day[i]-1]['end']
                        stint['data'].at[i:len(stint['data']), 'day'] = stint['data'].day[i]+1
                
                # Final step, perform the addition of time
                stint['data'].at[i, 'time'] = stint['data'].at[i-1, 'time'] + d_time
                stint['data'].at[i, 'time_unix'] = stint['data'].at[i, 'time'].timestamp()
                stint['data'].at[i, 'd_time'] = d_time
                stint['data'].at[i, 'd_timeDriving'] = d_timeDriving
                
                # Backfill i=0
                if i == 1:
                    stint['data'].at[0, 'time_unix'] = stint['data'].at[0, 'time'].timestamp()
                
            
            ### CALCULATE POSITION OF SUN ###
            
            # Create the location object
            l = Location()
            l.name = ''
            l.region = ''
            l.latitude = stint['data']['latitude'][i]
            l.longitude = stint['data']['longitude'][i]
            l.timezone = self.settings['time']['timezone']['name']
            l.elevation = 0
            
            stint['data'].at[i, 'solar__sunElevationAngle'] = l.solar_elevation(stint['data']['time'][i].to_pydatetime())
            stint['data'].at[i, 'solar__sunAzimuthAngle'] = l.solar_azimuth(stint['data']['time'][i].to_pydatetime())
            
                
        self.calculateArrivalDelta(stint)
    
    def getWeather(self, stint):
        # Fetch the weather conditions for every point on the route at the specified time
        
        if (not stint['isSensitivities']) & (stint['interation']%self.settings['weather']['iterationsPerEvaluation']==1):
            
            # Default values
            self.updateCol(stint['data'], 'weather__airTemp', 30)
            self.updateCol(stint['data'], 'weather__airPressure', 101250)
            self.updateCol(stint['data'], 'weather__humidity', 0)
            self.updateCol(stint['data'], 'weather__windSpeed', 0)
            self.updateCol(stint['data'], 'weather__windDirection', 0)
            self.updateCol(stint['data'], 'weather__windHeading', 180)
            self.updateCol(stint['data'], 'weather__airDensity', 1.225)
            self.updateCol(stint['data'], 'weather__cloudCover', 0.0)
            
            if self.settings['weather']['fromMongo']:
                
                d_query_norm = (stint['data'].distance.to_numpy() - self.weather['d_min'])/self.weather['d_range']
                t_query_norm = (stint['data'].time.astype(np.int64).to_numpy() * 1E-9 - self.weather['t_min'])/self.weather['t_range']
                
                print('Weather interp...')
                
                self.getWeather_interpolate(stint['data'], self.weather['d_norm'], self.weather['t_norm'], self.weather['airTemp'], d_query_norm, t_query_norm, 'weather__airTemp')
                self.getWeather_interpolate(stint['data'], self.weather['d_norm'], self.weather['t_norm'], self.weather['airPressure'], d_query_norm, t_query_norm, 'weather__airPressure')
                self.getWeather_interpolate(stint['data'], self.weather['d_norm'], self.weather['t_norm'], self.weather['humidity'], d_query_norm, t_query_norm, 'weather__humidity')
                self.getWeather_interpolate(stint['data'], self.weather['d_norm'], self.weather['t_norm'], self.weather['windSpeed'], d_query_norm, t_query_norm, 'weather__windSpeed')
                self.getWeather_interpolate(stint['data'], self.weather['d_norm'], self.weather['t_norm'], self.weather['windGust'], d_query_norm, t_query_norm, 'weather__windGust')
                self.getWeather_interpolate(stint['data'], self.weather['d_norm'], self.weather['t_norm'], self.weather['windDirection'], d_query_norm, t_query_norm, 'weather__windDirection')
                self.getWeather_interpolate(stint['data'], self.weather['d_norm'], self.weather['t_norm'], self.weather['windHeading'], d_query_norm, t_query_norm, 'weather__windHeading')
                self.getWeather_interpolate(stint['data'], self.weather['d_norm'], self.weather['t_norm'], self.weather['cloudCover'], d_query_norm, t_query_norm, 'weather__cloudCover')
                self.getWeather_interpolate(stint['data'], self.weather['d_norm'], self.weather['t_norm'], self.weather['precipProbability'], d_query_norm, t_query_norm, 'weather__precipProbability')
                self.getWeather_interpolate(stint['data'], self.weather['d_norm'], self.weather['t_norm'], self.weather['precipIntensity'], d_query_norm, t_query_norm, 'weather__precipIntensity')
                
                print('Weather interp... Complete')
                
            elif self.settings['weather']['fromCsv']:
                # Normalise the distance and time so that the interpolation algorithm is well conditioned
                d = self.weatherData.distance.to_numpy()
                d_min = min(d)
                d_max = max(d)
                d_range = d_max - d_min
                d_norm = (d - d_min)/d_range
                
                t = self.weatherData.datetime.astype(np.int64).to_numpy()
                t_min = min(t)
                t_max = max(t)
                t_range = t_max - t_min
                t_norm = (t - t_min)/t_range
                
                d_query_norm = (stint['data'].distance.to_numpy() - d_min)/d_range
                t_query_norm = (stint['data'].time.astype(np.int64).to_numpy() - t_min)/t_range
                
                # Interpolate the data for each quantity of interest
                self.getWeather_interpolate(stint['data'], d_norm, t_norm, self.weatherData.airTemp.to_numpy(), d_query_norm, t_query_norm, 'weather__airTemp')
                self.getWeather_interpolate(stint['data'], d_norm, t_norm, self.weatherData.airPressure.to_numpy(), d_query_norm, t_query_norm, 'weather__airPressure')
                self.getWeather_interpolate(stint['data'], d_norm, t_norm, self.weatherData.humidity.to_numpy(), d_query_norm, t_query_norm, 'weather__humidity')
                self.getWeather_interpolate(stint['data'], d_norm, t_norm, self.weatherData.windSpeed.to_numpy(), d_query_norm, t_query_norm, 'weather__windSpeed')
                self.getWeather_interpolate(stint['data'], d_norm, t_norm, self.weatherData.windCompN.to_numpy(), d_query_norm, t_query_norm, 'weather__windCompN')
                self.getWeather_interpolate(stint['data'], d_norm, t_norm, self.weatherData.windCompE.to_numpy(), d_query_norm, t_query_norm, 'weather__windCompE')
                
                # Change limits of direction to 0-360
                windDirection = self.rad2deg*np.arctan2(stint['data']['weather__windCompE'].to_numpy(), stint['data']['weather__windCompN'].to_numpy())
                windDirectionClean = windDirection + (windDirection<0)*360
                windHeading = windDirection + 180
                self.updateCol(stint['data'], 'weather__windDirection', windDirectionClean )
                self.updateCol(stint['data'], 'weather__windHeading', windHeading )
                
            if self.settings['weather']['fromMongo'] or self.settings['weather']['fromCsv']:
                ### CALCULATE OTHER PARAMETERS ###
                # Calculate air density
    #            ρ = 1 / v
    #                = (p / Ra T) (1 + x) / (1 + x Rw / Ra)
                humidity = stint['data']['weather__humidity'].to_numpy()
                rho_dryAir = stint['data']['weather__airPressure'].to_numpy() /(self.Ra * (self.C2K + stint['data']['weather__airTemp'].to_numpy()))
                rho = rho_dryAir * (1+humidity) / (1 + humidity * self.Rw/self.Ra)
                
                self.updateCol(stint['data'], 'weather__airDensity', rho)
            
            
        
    def getWeather_interpolate(self, df, d, t, values, d_query, t_query, paramName):
        
        interpolatedValues = griddata((d, t), values, (d_query, t_query), method='linear')
        self.updateCol(df, paramName, interpolatedValues)
        
    def getSolarProjectedArea(self, xRotation, yRotation):
        
        griddata((self.solarXRotation, self.solarYRotation), self.solarAreaRatio, (xRotation, yRotation), method='linear')
    
    def calculateAero(self, stint):
        CdA = self.settings['aero']['CdA']
        
        # Calulate wind effect
        self.updateCol(stint['data'], 'aero__headingDeltaCarWind', stint['data']['weather__windHeading'] - stint['data']['heading'])
        self.updateCol(stint['data'], 'aero__vTailwind', (stint['data']['weather__windSpeed'].to_numpy() * np.cos(stint['data']['aero__headingDeltaCarWind'].to_numpy() * self.deg2rad) ) )
        self.updateCol(stint['data'], 'aero__vCrossWind', (stint['data']['weather__windSpeed'].to_numpy() * np.sin(stint['data']['aero__headingDeltaCarWind'].to_numpy() * self.deg2rad) ) )
        self.updateCol(stint['data'], 'aero__airSpeedForward', (stint['data']['speed'].to_numpy() - stint['data']['aero__vTailwind'].to_numpy() ) )
        
        # Calculate forces
        self.updateCol(stint['data'], 'aero__dragForce', CdA*0.5*stint['data']['weather__airDensity'].to_numpy()*(stint['data']['aero__airSpeedForward'].to_numpy()*self.kph2ms)**2)
        
        
        self.updateCol(stint['data'], 'aero__dragPower', stint['data'].aero__dragForce*stint['data'].speed*self.kph2ms)
        self.updateCol(stint['data'], 'aero__d_dragEnergy', 0)
        self.updateCol(stint['data'], 'aero__dragEnergy', 0)
        
        for i in range(1, len(stint['data'])):
            averageDragForce = 0.5*stint['data'].aero__dragForce[i] + 0.5*stint['data'].aero__dragForce[i-1]
            stint['data'].at[i, 'aero__d_dragEnergy'] = averageDragForce*stint['data'].d_distance[i]*1000
            stint['data'].at[i, 'aero__dragEnergy'] = stint['data'].at[i-1, 'aero__dragEnergy'] + stint['data'].at[i, 'aero__d_dragEnergy']
    
    def calculateMech(self, stint):
        
        ### TYRES ###
        Crr = self.settings['tyres']['Crr']
        
        carMass = self.settings['car']['mass']
        carWeight = carMass*self.g
        
        self.updateCol(stint['data'], 'car__ForceNormal', carWeight*np.cos(stint['data'].inclination_angle.to_numpy()))
        self.updateCol(stint['data'], 'car__ForceLongitudinal', carWeight*np.sin(stint['data'].inclination_angle.to_numpy()))
        
        self.updateCol(stint['data'], 'mech__tyreRollingResistanceForce', Crr*stint['data'].car__ForceNormal)
        self.updateCol(stint['data'], 'mech__tyreRollingResistancePower', stint['data'].mech__tyreRollingResistanceForce*stint['data'].speed*self.kph2ms)
        self.updateCol(stint['data'], 'mech__d_tyreRollingResistanceEnergy', 0)
        self.updateCol(stint['data'], 'mech__tyreRollingResistanceEnergy', 0)
        
        ### CHASSIS ###
        self.updateCol(stint['data'], 'mech__chassisRollingResistanceForce', 0)
        self.updateCol(stint['data'], 'mech__chassisRollingResistancePower', stint['data'].mech__chassisRollingResistanceForce*stint['data'].speed*self.kph2ms)
        self.updateCol(stint['data'], 'mech__d_chassisRollingResistanceEnergy', 0)
        self.updateCol(stint['data'], 'mech__chassisRollingResistanceEnergy', 0)
        
        ### GRAVITY ###
        self.updateCol(stint['data'], 'mech__gravityResistanceForce', self.settings['car']['mass'] * self.g * np.sin(stint['data']['inclination_angle'].to_numpy()) )
        self.updateCol(stint['data'], 'mech__gravityRollingResistancePower', stint['data'].mech__gravityResistanceForce*stint['data'].speed*self.kph2ms)
        self.updateCol(stint['data'], 'mech__d_gravityRollingResistanceEnergy', 0)
        self.updateCol(stint['data'], 'mech__gravityRollingResistanceEnergy', 0)
        
        ### TOTAL ###
        self.updateCol(stint['data'], 'mech__totalResistiveForce', stint['data'].mech__tyreRollingResistanceForce + stint['data'].mech__chassisRollingResistanceForce + stint['data'].mech__gravityResistanceForce)
        self.updateCol(stint['data'], 'mech__totalResistivePower', stint['data'].mech__tyreRollingResistanceForce*stint['data'].speed*self.kph2ms)
        self.updateCol(stint['data'], 'mech__d_totalResistiveEnergy', 0)
        self.updateCol(stint['data'], 'mech__totalResistiveEnergy', 0)
        
        ### LOOP ###
        for i in range(1, len(stint['data'])):
            mech__tyreRollingResistanceForce_avg = 0.5*stint['data'].mech__tyreRollingResistanceForce[i] + 0.5*stint['data'].mech__tyreRollingResistanceForce[i-1]
            stint['data'].at[i, 'mech__d_tyreRollingResistanceEnergy'] = mech__tyreRollingResistanceForce_avg*stint['data'].d_distance[i]*1000
            stint['data'].at[i, 'mech__tyreRollingResistanceEnergy'] = stint['data'].at[i-1, 'mech__tyreRollingResistanceEnergy'] + stint['data'].at[i, 'mech__d_tyreRollingResistanceEnergy']
            
            mech__chassisRollingResistanceForce_avg = 0.5*stint['data'].mech__chassisRollingResistanceForce[i] + 0.5*stint['data'].mech__chassisRollingResistanceForce[i-1]
            stint['data'].at[i, 'mech__d_chassisRollingResistanceEnergy'] = mech__chassisRollingResistanceForce_avg*stint['data'].d_distance[i]*1000
            stint['data'].at[i, 'mech__chassisRollingResistanceEnergy'] = stint['data'].at[i-1, 'mech__chassisRollingResistanceEnergy'] + stint['data'].at[i, 'mech__d_chassisRollingResistanceEnergy']
            
            mech__gravityResistanceForce_avg = 0.5*stint['data'].mech__gravityResistanceForce[i] + 0.5*stint['data'].mech__gravityResistanceForce[i-1]
            stint['data'].at[i, 'mech__d_gravityRollingResistanceEnergy'] = mech__gravityResistanceForce_avg*stint['data'].d_distance[i]*1000
            stint['data'].at[i, 'mech__gravitysRollingResistanceEnergy'] = stint['data'].at[i-1, 'mech__gravityRollingResistanceEnergy'] + stint['data'].at[i, 'mech__d_gravityRollingResistanceEnergy']
            
            mech__totalResistiveForce_avg = 0.5*stint['data'].mech__totalResistiveForce[i] + 0.5*stint['data'].mech__totalResistiveForce[i-1]
            stint['data'].at[i, 'mech__d_totalResistiveEnergy'] = mech__totalResistiveForce_avg*stint['data'].d_distance[i]*1000
            stint['data'].at[i, 'mech__totalResistiveEnergy'] = stint['data'].at[i-1, 'mech__totalResistiveEnergy'] + stint['data'].at[i, 'mech__d_totalResistiveEnergy']
        
    def calculateElec(self, stint):
        
        ### MOTOR ###
        # Torque and speed
        rollingRadius = self.settings['tyres']['rollingRadius']
        NMotors = self.settings['car']['NMotors']
        
        resistiveForcesCombined = stint['data']['aero__dragForce'].to_numpy() + stint['data']['mech__totalResistiveForce'].to_numpy()
        
        motorTorque = rollingRadius * resistiveForcesCombined / NMotors
        motorSpeed = stint['data']['speed'].to_numpy() * self.kph2ms / rollingRadius
        motorPowerTractive = motorTorque * motorSpeed
        
        self.updateCol(stint['data'], 'elec__motorTorque', motorTorque)
        self.updateCol(stint['data'], 'elec__motorSpeed', motorSpeed)
        self.updateCol(stint['data'], 'elec__motorSpeedRPM', motorSpeed*self.rads2RPM)
        self.updateCol(stint['data'], 'elec__motorPowerTractive', motorPowerTractive)
        
        # Temperature
        for i in range(0, len(stint['data'])):
            Tw = 323 # Approx winding temp, initial condition
            
            Tw_err = np.inf
            
            while Tw_err > 1 :
                Tm = 0.5*(stint['data']['weather__airTemp'][i]+self.C2K + Tw) # Magnet temp
                B = 1.32-1.2E-3 * (Tm - 293) # Magnet remanence
                i_rms = 0.561*B*stint['data']['elec__motorTorque'][i] # RMS per phase motor current
                R = 0.0575 * (1 + 0.0039*(Tw - 293)) # Per phase motor winding resistance
                Pc = 3*i_rms**2*R # Total motor winding i2R copper loss
                Pe = (9.602E-6 * (B*stint['data']['elec__motorSpeed'][i])**2) / R # Total motor eddy current loss
                
                Tw_new = 0.455*(Pc + Pe) + stint['data']['weather__airTemp'][i]+self.C2K # New estimate for motor winding temperature
                Tw_err = np.abs(Tw_new - Tw)
                Tw = Tw_new
            
            stint['data'].at[i, 'elec__motorTempWinding'] = Tw
            stint['data'].at[i, 'elec__motorTempMagnet'] = Tm
            stint['data'].at[i, 'elec__motorMagnetRemanence'] = B
            stint['data'].at[i, 'elec__motorCurrentPerPhase'] = i_rms
            stint['data'].at[i, 'elec__motorResistanceWinding'] = R
            stint['data'].at[i, 'elec__motorPowerWinding'] = Pc
            stint['data'].at[i, 'elec__motorPowerEddyCurrent'] = Pe
            stint['data'].at[i, 'elec__motorPowerLossTotal'] = Pc + Pe
        
        powerMotorTotal = motorPowerTractive + stint['data']['elec__motorPowerLossTotal'].to_numpy()
        self.updateCol(stint['data'], 'elec__motorPowerTotal', powerMotorTotal)
        
        ### MOTOR CONTROLLER ###
        
        efficiencyMotorController = griddata((self.efficiencyMotorController.motorSpeed, self.efficiencyMotorController.motorTorque), self.efficiencyMotorController.efficiency, (stint['data']['elec__motorSpeed'].to_numpy(), stint['data']['elec__motorTorque'].to_numpy()), method='linear')
        motorControllerPowerLoss = (1 / efficiencyMotorController - 1) * powerMotorTotal
        
        self.updateCol(stint['data'], 'elec__efficiencyMotorController', efficiencyMotorController)
        self.updateCol(stint['data'], 'elec__motorControllerPowerLoss', motorControllerPowerLoss)
        
        
        ### BATTERY ###
        
        powerDemand = powerMotorTotal + motorControllerPowerLoss
        
        cellVoltage = griddata(self.batteryCellDischargeCurve.rSOC, self.batteryCellDischargeCurve.voltage, stint['data']['car__rSOC'].to_numpy(), method='linear')
        packVoltage = cellVoltage * self.settings['battery']['NCellsSeries']
        
        cellResistance = self.settings['battery']['resistanceInternalCell']
        packResistance = cellResistance / self.settings['battery']['NCellsParallel'] * self.settings['battery']['NCellsSeries']
        
        packCurrent = powerDemand / packVoltage
        
        packPowerLoss = packCurrent**2 * packResistance
        
        self.updateCol(stint['data'], 'elec__batteryCellVoltage', cellVoltage)
        self.updateCol(stint['data'], 'elec__batteryPackVoltage', packVoltage)
        self.updateCol(stint['data'], 'elec__batteryPackCurrent', packCurrent)
        self.updateCol(stint['data'], 'elec__batteryPackPowerLoss', packPowerLoss)
        
        ### TOTAL ###
        
        self.updateCol(stint['data'], 'elec__totalLossesPower', Pc + Pe + motorControllerPowerLoss + packPowerLoss)
        self.updateCol(stint['data'], 'elec__d_totalLossesEnergy', 0)
        self.updateCol(stint['data'], 'elec__totalLossesEnergy', 0)
        
        for i in range(1, len(stint['data'])):
            
            power_avg = 0.5*stint['data'].elec__totalLossesPower[i] + 0.5*stint['data'].elec__totalLossesPower[i-1]
            
            stint['data'].at[i, 'elec__d_totalLossesEnergy'] = power_avg*stint['data']['d_timeDriving'][i].seconds
            stint['data'].at[i, 'elec__totalLossesEnergy'] = stint['data'].at[i-1, 'elec__totalLossesEnergy'] + stint['data'].at[i, 'elec__d_totalLossesEnergy']
        
    def calculateSolar(self, stint):
        sunAzimuthRelativeCar = (stint['data']['solar__sunAzimuthAngle'].to_numpy() - stint['data']['heading'].to_numpy()) * self.deg2rad
#        sunAzimuthRelativeCar[sunAzimuthRelativeCar<0] = sunAzimuthRelativeCar[sunAzimuthRelativeCar<0] + math.pi
        sunElevation = stint['data']['solar__sunElevationAngle'].to_numpy() * self.deg2rad
        
        temp = np.arctan( np.sin(sunElevation) / (np.cos(sunElevation) *  np.sin(sunAzimuthRelativeCar)) ) * self.rad2deg
        rotationX = -np.sign(temp)*90 + temp
        
        temp2 = np.arctan( np.sin(sunElevation) / (np.cos(sunElevation) *  np.cos(sunAzimuthRelativeCar)) ) * self.rad2deg
        rotationY = -np.sign(temp2)*90 + temp2
        
        self.updateCol(stint['data'], 'solar__sunAzimuthRelativeCar', sunAzimuthRelativeCar*self.rad2deg)
        self.updateCol(stint['data'], 'solar__rotationX', rotationX)
        self.updateCol(stint['data'], 'solar__rotationY', rotationY)
        
        projectedAreaRatio = griddata((self.solarXRotation, self.solarYRotation), self.solarAreaRatio, (np.abs(rotationX), rotationY), method='linear')
        projectedArea = projectedAreaRatio * self.settings['solar']['NCells'] * self.settings['solar']['areaPerCell'] * self.settings['solar']['ratioProjectedFlat']
        
        self.updateCol(stint['data'], 'solar__projectedAreaRatio', projectedAreaRatio)
        self.updateCol(stint['data'], 'solar__projectedArea', projectedArea)
        
        cloudCover = stint['data']['weather__cloudCover'].to_numpy()
        irradianceNominal = self.settings['solar']['irradianceNominal']
        
        powerIncidentOnArray = irradianceNominal * (1 - cloudCover) * projectedArea
        powerCapturedArray = powerIncidentOnArray * self.settings['solar']['efficiencyEncapsulation'] * self.settings['solar']['efficiencyCell']
        
        self.updateCol(stint['data'], 'solar__powerIncidentOnArray', powerIncidentOnArray)
        self.updateCol(stint['data'], 'solar__powerCapturedArray', powerCapturedArray)
        
        self.updateCol(stint['data'], 'solar__d_energyCapturedArray', 0)
        self.updateCol(stint['data'], 'solar__energyCapturedArray', 0)
        for i in range(1, len(stint['data'])):
            power_avg = 0.5*stint['data'].solar__powerCapturedArray[i] + 0.5*stint['data'].solar__powerCapturedArray[i-1]
            delta_time = stint['data'].time[i] - stint['data'].time[i-1]
            
            stint['data'].at[i, 'solar__d_energyCapturedArray'] = power_avg*delta_time.seconds
            stint['data'].at[i, 'solar__energyCapturedArray'] = stint['data'].at[i-1, 'solar__energyCapturedArray'] + stint['data'].at[i, 'solar__d_energyCapturedArray']
            
        
    def calculateEnergy(self, stint):
        
        self.updateCol(stint['data'], 'car__powerUsed', stint['data'].aero__dragPower + stint['data'].mech__totalResistivePower + stint['data'].elec__totalLossesPower)
        self.updateCol(stint['data'], 'car__d_energyUsed', stint['data'].aero__d_dragEnergy + stint['data'].mech__d_totalResistiveEnergy + stint['data'].elec__d_totalLossesEnergy)
        self.updateCol(stint['data'], 'car__energyUsed', stint['data'].aero__dragEnergy + stint['data'].mech__totalResistiveEnergy + stint['data'].elec__totalLossesEnergy)
        
        self.updateCol(stint['data'], 'car__SOC_Delta', stint['data'].solar__energyCapturedArray - stint['data'].car__energyUsed)
        self.updateCol(stint['data'], 'car__powerDelta', stint['data'].solar__powerCapturedArray - stint['data'].car__powerUsed)
        
        self.updateCol(stint['data'], 'car__SOC', stint['SOCInitial'] + stint['data'].car__SOC_Delta.to_numpy())
        self.updateCol(stint['data'], 'car__rSOC', stint['data'].car__SOC.to_numpy() / self.settings['battery']['capacity'])
    
    def calculateSensitivities(self, stint):
        
        speedPerturbation = 0.1
        
        # Make a copy of the stint
        stintCopy = copy.deepcopy(stint)
        stintCopy['isSensitivities'] = True
        
        # Perturb the speed
        stintCopy['data'].speed = stintCopy['data'].speed + speedPerturbation
        
        # Run the model with the perturned speed
        self.runModels(stintCopy)
        
        # Calculate effect on power
        self.updateCol(stint['data'], 'sens__powerPerKph', (stintCopy['data'].car__powerUsed - stint['data'].car__powerUsed)/speedPerturbation)
        self.updateCol(stint['data'], 'sens__energyPerKph', (stintCopy['data'].car__d_energyUsed - stint['data'].car__d_energyUsed)/speedPerturbation)
        
    def adjustSpeed(self, stint):
        changesMade = ''
        
        # Correct speeds if they violate constraint limit
        self.updateCol(stint['data'], 'speed', pd.DataFrame([stint['data'].speed, stint['data'].speedMin]).max())
        self.updateCol(stint['data'], 'speed', pd.DataFrame([stint['data'].speed, stint['data'].speedMax]).min())
        
        # Determine if speed is at the constraint limit
        self.updateCol(stint['data'], 'onSpeedMin', stint['data'].speed - stint['data'].speedMin <= 0)
        self.updateCol(stint['data'], 'onSpeedMax', stint['data'].speed - stint['data'].speedMax >= 0)        
        
        self.updateCol(stint['data'], 'sens_powerPerKphDeltaToMax', stint['data'].sens__powerPerKph - stint['data'].loc[~stint['data'].onSpeedMax, ['sens__powerPerKph']].max().to_list())
        self.updateCol(stint['data'], 'sens_powerPerKphDeltaToMin', stint['data'].sens__powerPerKph - stint['data'].loc[~stint['data'].onSpeedMin, ['sens__powerPerKph']].min().to_list())
        
        # Gate the power sensitivity to speed by whether it's still possible to change the speed there
        self.updateCol(stint['data'], 'sens_powerPerKphDeltaToMax_gated', stint['data'].sens_powerPerKphDeltaToMax * (~stint['data'].onSpeedMax).astype(int))
        self.updateCol(stint['data'], 'sens_powerPerKphDeltaToMin_gated', stint['data'].sens_powerPerKphDeltaToMin * (~stint['data'].onSpeedMin).astype(int))
        
        # Calculate weightings to use for deciding how much speed to add or subtract at each location
        if stint['data'].sens_powerPerKphDeltaToMax_gated.sum() != 0:
            weightAdd = stint['data'].sens_powerPerKphDeltaToMax_gated / stint['data'].sens_powerPerKphDeltaToMax_gated.sum()
            weightSubtract =  stint['data'].sens_powerPerKphDeltaToMin_gated / stint['data'].sens_powerPerKphDeltaToMin_gated.sum()
        else:
            weightAdd = 1/len(stint['data']) + stint['data'].sens_powerPerKphDeltaToMax_gated*0
            weightSubtract = 1/len(stint['data']) + stint['data'].sens_powerPerKphDeltaToMax_gated*0
        
        # Adjust the weightings so that only the most weighted points get adjusted
        convAggro = self.settings['simulation']['convergenceAggressiveness']
        weightAddProcessed = weightAdd * (weightAdd >= (weightAdd.max() - (weightAdd.max()-weightAdd.min())*convAggro))
        weightSubtractProcessed = weightSubtract * (weightSubtract >= (weightSubtract.max() - (weightSubtract.max()-weightSubtract.min())*convAggro))
        
        print('weightAddProcessed: {} entries'.format((weightAddProcessed>0).sum()))
        print('weightSubtractProcessed: {} entries'.format((weightSubtractProcessed>0).sum()))
        
        self.updateCol(stint['data'], 'sens_powerPerKph_weightAdd', weightAddProcessed)
        self.updateCol(stint['data'], 'sens_powerPerKph_weightSubtract', weightSubtractProcessed)
        
        self.calculateArrivalDelta(stint)
        
        stepSize = np.nan
        
        # Check if we are too slow to achieve the arrival time
        if stint['arrivalTimeDelta'] > self.settings['simulation']['arrivalTimeTolerance'] :
            # Increase speed at cheap locations
            stepSize = max(min(10,0.6*stint['data'].sens_powerPerKphDeltaToMin_gated.max()), min(2, 0.001*stint['arrivalTimeDelta']**2), self.settings['simulation']['minStepSizeSpeedAdd'])
            
            # Set new speed
            self.updateCol(stint['data'], 'speed', stint['data'].speed + stepSize*stint['data'].sens_powerPerKph_weightAdd)
            
            # Apply speed constraints
            self.updateCol(stint['data'], 'speed', pd.DataFrame([stint['data'].speed, stint['data'].speedMin]).max())
            self.updateCol(stint['data'], 'speed', pd.DataFrame([stint['data'].speed, stint['data'].speedMax]).min())
            
            changesMade = '+Speed'
            
        elif (stint['arrivalTimeDelta'] < -self.settings['simulation']['arrivalTimeTolerance']) | (stint['data'].sens_powerPerKphDeltaToMin_gated.max() > self.settings['simulation']['powerSensitivityTolerance']):
            # Decrease speed at expensive locations
            stepSize = max(min(10,stint['data'].sens_powerPerKphDeltaToMin_gated.max()), min(10,-stint['arrivalTimeDelta']), self.settings['simulation']['minStepSizeSpeedSubtract'])
            
            # Set new speed
            if stint['arrivalTimeDelta'] > 0:
                self.updateCol(stint['data'], 'speed', stint['data'].speed + stepSize*stint['data'].sens_powerPerKph_weightAdd)
                changesMade = '+Speed'
            else:
                self.updateCol(stint['data'], 'speed', stint['data'].speed - stepSize*stint['data'].sens_powerPerKph_weightSubtract)
                changesMade = '-Speed'
            
            # Apply speed constraints
            self.updateCol(stint['data'], 'speed', pd.DataFrame([stint['data'].speed, stint['data'].speedMin]).max())
            self.updateCol(stint['data'], 'speed', pd.DataFrame([stint['data'].speed, stint['data'].speedMax]).min())
            
            
        
        print('stepSize: {}'.format(stepSize))
        return changesMade
    
    def calculateArrivalDelta(self, stint):
        stint['arrivalTimeDelta'] = (stint['data'].time.iloc[-1] - stint['arrivalTime']).seconds
        if stint['data'].time.iloc[-1] < stint['arrivalTime']:
            stint['arrivalTimeDelta'] = -(stint['arrivalTime'] - stint['data'].time.iloc[-1]).seconds
        
    def combineStints(self):
        initialised = False
        for iStint in range(0, self.NStints):
            stint = self.stints[iStint]
            
            if 'data' in stint:
                if not initialised:
                    self.data = stint['data']
                    initialised = True
                else:
                    self.data = self.data.append(stint['data'], ignore_index=True)
                
    def writeOutput(self):

        outputFolder = '{}\\..\\Cases'.format(os.getcwd())
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
            
        self.data.to_csv('{}\\{}.csv'.format(outputFolder, self.settings['meta']['name']))
        
    def updateCol(self, df, colName, colValues):
        if colName not in df.columns:
            df.insert(len(df.columns), colName, colValues)
        else:
            df[colName] = colValues
            
    def unique(self, list1):
        unique_list = []
        list1.sort()
        for x in list1: 
            if x not in unique_list: 
                unique_list.append(x) 
        return unique_list