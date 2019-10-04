import pandas as pd
import yaml
import datetime
import dateutil
import math
import numpy as np
from scipy.interpolate import griddata
import os
import copy

class Simulation:
    
    kph2ms = 1/3.6
    ms2kph = 3.6
    rad2deg = 180.0/math.pi
    deg2rad = math.pi/180.0
    g = 9.80665
    R = 287.6
    C2K = 273.15
    K2C = -273.15
    
    def __init__(self, settingsPath):

        # Load settings
        self.settingsPath = settingsPath
        with open(settingsPath, 'r') as stream:
            try:
                self.settings = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        self.initData()
        
    def initData(self):
        self.getRoute()
        self.getWeatherData()
        
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
            
    def getWeatherData(self):
        
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
                
        self.weatherDistances = self.unique(self.weatherData.distance.tolist())
            
    def splitStints(self):
        # Set the stints
        self.stints = self.settings['route']['stints']
        self.NStints = len(self.stints)
        
        for iStint in range(0, self.NStints):
            stintNumber = iStint+1
            
            startDistance = -1
            end_distance = -1
                
            startDistance = self.locations[self.stints[iStint]['startLocation']]['distance']
            end_distance = self.locations[self.stints[iStint]['endLocation']]['distance']
            
            # Find the section of the table that contains the current stint
            inStint = ((self.data.distance > startDistance) & (self.data.distance <= end_distance)).values.tolist()
            startIndex = inStint.index(True) - 1
            endIndex = len(inStint) - inStint[::-1].index(True) - 1
            
            # Get the data that belongs to this stint
            self.stints[iStint]['data'] = self.data.iloc[startIndex:endIndex+1, :]
            
            # Reset the index
            self.stints[iStint]['data'] = self.stints[iStint]['data'].reset_index(drop=True)
            
            # Add some meta data
            self.stints[iStint]['nStint'] = stintNumber
            self.stints[iStint]['startDistance'] = startDistance
            self.stints[iStint]['end_distance'] = end_distance
            self.stints[iStint]['stintLength'] = end_distance - startDistance
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
            if self.settings['initialConditions']['time'] > self.settings['time']['days'][self.stints[iStint]['startDay']-1]['start']:
                self.stints[iStint]['startTime'] = self.settings['initialConditions']['time']
            else:
                self.stints[iStint]['startTime'] = self.settings['time']['days'][self.stints[iStint]['startDay']-1]['start']
            
            # Find the end day
            for iDay in range(0, len(self.settings['time']['days'])):
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
                    self.stints[iStint]['availableTime'] += self.settings['time']['days'][iDay]['end'] - self.stints[iStint]['startTime']
                    
                elif iDay == self.stints[iStint]['endDay']-1:
                    # Last day of stint
                    self.stints[iStint]['availableTime'] += self.stints[iStint]['arrivalTime'] - self.settings['time']['days'][iDay]['start']
                    
                else:
                    # Day between start and end days (not applicable for 2019)
                    self.stints[iStint]['availableTime'] += self.settings['time']['days'][iDay]['end'] - self.settings['time']['days'][iDay]['start']
            
            self.stints[iStint]['availableTime'] -= self.stints[iStint]['NControlStops']*datetime.timedelta(minutes=self.settings['time']['controlStops']['duration'])
            
            
            # Initialise the speed
            self.stints[iStint]['averageSpeed'] = self.stints[iStint]['stintLength'] / (self.stints[iStint]['availableTime'].seconds/3600)
            self.updateCol(self.stints[iStint]['data'], 'speed', self.stints[iStint]['averageSpeed'])
            self.updateCol(self.stints[iStint]['data'], 'speedms', self.stints[iStint]['averageSpeed']*self.kph2ms)
            
    def runModels(self, stint):
        self.getWeather(stint)
        self.calculateAero(stint)
        self.calculateMech(stint)
        self.calculateElec(stint)
        self.calculateEnergy(stint)
    
    def calculateTime(self, stint):
        # Calculates the time using the car speed as input
        self.updateCol(stint['data'], 'speedms', stint['data']['speed']*self.kph2ms)
        
        self.updateCol(stint['data'], 'time', stint['startTime'])
        self.updateCol(stint['data'], 'day', stint['startDay'])
        self.updateCol(stint['data'], 'time_unix', stint['startTime'].timestamp())
        self.updateCol(stint['data'], 'd_time', 0)
        self.updateCol(stint['data'], 'd_timeDriving', 0)
        
        for i in range(1, len(stint['data'])):
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
                
        self.calculateArrivalDelta(stint)
    
    def getWeather(self, stint):
        
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
        
        ### CALCULATE OTHER PARAMETERS ###
        # Calculate air density
        self.updateCol(stint['data'], 'weather__airDensity', stint['data']['weather__airPressure'].to_numpy() / (self.R * (self.C2K + stint['data']['weather__airTemp'].to_numpy())) )
        
        
    def getWeather_interpolate(self, df, d, t, values, d_query, t_query, paramName):
        
        interpolatedValues = griddata((d, t), values, (d_query, t_query), method='linear')
        self.updateCol(df, paramName, interpolatedValues)
    
    def calculateAero(self, stint):
        CdA = self.settings['aero']['CdA']
        
        # Calulate wind effect
        self.updateCol(stint['data'], 'aero__headingDeltaCarWind', stint['data']['weather__windHeading'] - stint['data']['heading'])
        self.updateCol(stint['data'], 'aero__vTailwind', (stint['data']['weather__windSpeed'].to_numpy() * np.cos(stint['data']['aero__headingDeltaCarWind'].to_numpy() * self.deg2rad) ) )
        self.updateCol(stint['data'], 'aero__vCrossWind', (stint['data']['weather__windSpeed'].to_numpy() * np.sin(stint['data']['aero__headingDeltaCarWind'].to_numpy() * self.deg2rad) ) )
        self.updateCol(stint['data'], 'aero__airSpeedForward', (stint['data']['speed'].to_numpy() - stint['data']['aero__vTailwind'].to_numpy() ) )
        
        # Calculate forces
        self.updateCol(stint['data'], 'aero__dragForce', CdA*0.5*stint['data']['weather__airDensity'].to_numpy()*(stint['data']['aero__airSpeedForward'].to_numpy()*self.kph2ms)**2)
        
        
        self.updateCol(stint['data'], 'aero__dragPower', stint['data'].aero__dragForce*stint['data'].speedms)
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
        self.updateCol(stint['data'], 'mech__tyreRollingResistancePower', stint['data'].mech__tyreRollingResistanceForce*stint['data'].speedms)
        self.updateCol(stint['data'], 'mech__d_tyreRollingResistanceEnergy', 0)
        self.updateCol(stint['data'], 'mech__tyreRollingResistanceEnergy', 0)
        
        ### GENERAL ###
        self.updateCol(stint['data'], 'mech__chassisRollingResistanceForce', 0)
        self.updateCol(stint['data'], 'mech__chassisRollingResistancePower', stint['data'].mech__chassisRollingResistanceForce*stint['data'].speedms)
        self.updateCol(stint['data'], 'mech__d_chassisRollingResistanceEnergy', 0)
        self.updateCol(stint['data'], 'mech__chassisRollingResistanceEnergy', 0)
        
        ### TOTAL ###
        self.updateCol(stint['data'], 'mech__totalResistiveForce', Crr*stint['data'].car__ForceNormal)
        self.updateCol(stint['data'], 'mech__totalResistivePower', stint['data'].mech__tyreRollingResistanceForce*stint['data'].speedms)
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
            
            mech__totalResistiveForce_avg = 0.5*stint['data'].mech__totalResistiveForce[i] + 0.5*stint['data'].mech__totalResistiveForce[i-1]
            stint['data'].at[i, 'mech__d_totalResistiveEnergy'] = mech__totalResistiveForce_avg*stint['data'].d_distance[i]*1000
            stint['data'].at[i, 'mech__totalResistiveEnergy'] = stint['data'].at[i-1, 'mech__totalResistiveEnergy'] + stint['data'].at[i, 'mech__d_totalResistiveEnergy']
        
    def calculateElec(self, stint):
        
        # Fetch parameters
        rollingRadius = self.settings['tyres']['rollingRadius']
        NMotors = self.settings['car']['NMotors']
        
        
        resistiveForcesCombined = stint['data']['aero__dragForce'].to_numpy() + stint['data']['mech__totalResistiveForce'].to_numpy()
        
        motorTorque = resistiveForcesCombined / NMotors * rollingRadius
        
        self.updateCol(stint['data'], 'elec__motorTorque', motorTorque)
        
        
        
    def calculateEnergy(self, stint):
        
        self.updateCol(stint['data'], 'car__powerUsed', stint['data'].aero__dragPower + stint['data'].mech__totalResistivePower)
        self.updateCol(stint['data'], 'car__d_energyUsed', stint['data'].aero__d_dragEnergy + stint['data'].mech__d_totalResistiveEnergy)
        self.updateCol(stint['data'], 'car__energyUsed', stint['data'].aero__dragEnergy + stint['data'].mech__totalResistiveEnergy)
    
    def calculateSensitivities(self, stint):
        
        speedPerturbation = 0.1
        
        # Make a copy of the stint
        stintCopy = copy.deepcopy(stint)
        
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
            weightAdd = 1/len(stint['data'])
            weightSubtract = 1/len(stint['data'])
        
        # Adjust the weightings so that only the most weighted points get adjusted
        convAggro = self.settings['simulation']['convergenceAggressiveness']
        weightAddProcessed = weightAdd * (weightAdd > (weightAdd.max() - (weightAdd.max()-weightAdd.min())*convAggro))
        weightSubtractProcessed = weightSubtract * (weightSubtract > (weightSubtract.max() - (weightSubtract.max()-weightSubtract.min())*convAggro))
        
        print('weightAddProcessed: {} entries'.format((weightAddProcessed>0).sum()))
        print('weightSubtractProcessed: {} entries'.format((weightSubtractProcessed>0).sum()))
        
        self.updateCol(stint['data'], 'sens_powerPerKph_weightAdd', weightAddProcessed)
        self.updateCol(stint['data'], 'sens_powerPerKph_weightSubtract', weightSubtractProcessed)
        
        self.calculateArrivalDelta(stint)
        
        stepSize = max(min(10,stint['data'].sens_powerPerKphDeltaToMin_gated.max()*10), min(10,-stint['arrivalTimeDelta']), 0.1)
        print('stepSize: {}'.format(stepSize))
        
        # Check if we are too slow to achieve the arrival time
        if stint['arrivalTimeDelta'] > 0 :
            # Increase speed at cheap locations
#            stepSize = max(0.1*stint['arrivalTimeDelta'], 0.1)
            
            # Set new speed
            self.updateCol(stint['data'], 'speed', stint['data'].speed + stepSize*stint['data'].sens_powerPerKph_weightAdd)
            
            # Apply speed constraints
            self.updateCol(stint['data'], 'speed', pd.DataFrame([stint['data'].speed, stint['data'].speedMin]).max())
            self.updateCol(stint['data'], 'speed', pd.DataFrame([stint['data'].speed, stint['data'].speedMax]).min())
            
            changesMade = '+Speed'
            
        elif (stint['arrivalTimeDelta'] < -self.settings['simulation']['arrivalTimeTolerance']) | (stint['data'].sens_powerPerKphDeltaToMin_gated.max() > self.settings['simulation']['powerSensitivityTolerance']):
            # Decrease speed at expensive locations
#            stepSize = max(min(10,stint['data'].sens_powerPerKphDeltaToMin_gated.max()*10), min(10,-stint['arrivalTimeDelta']), 0.1)
            
            # Set new speed
#            self.updateCol(stint['data'], 'speed', stint['data'].speed + 0.1*stepSize*stint['data'].sens_powerPerKph_weightAdd)
            self.updateCol(stint['data'], 'speed', stint['data'].speed - stepSize*stint['data'].sens_powerPerKph_weightSubtract)
            
            
            # Apply speed constraints
            self.updateCol(stint['data'], 'speed', pd.DataFrame([stint['data'].speed, stint['data'].speedMin]).max())
            self.updateCol(stint['data'], 'speed', pd.DataFrame([stint['data'].speed, stint['data'].speedMax]).min())
            
            changesMade = '-Speed'
        
        return changesMade
    
    def calculateArrivalDelta(self, stint):
        stint['arrivalTimeDelta'] = (stint['data'].time.iloc[-1] - stint['arrivalTime']).seconds
        if stint['data'].time.iloc[-1] < stint['arrivalTime']:
            stint['arrivalTimeDelta'] = -(stint['arrivalTime'] - stint['data'].time.iloc[-1]).seconds
        
    def combineStints(self):
        for iStint in range(0, self.NStints):
            if iStint == 0:
                self.data = self.stints[iStint]['data']
            else:
                self.data = self.data.append(self.stints[iStint]['data'], ignore_index=True)
                
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