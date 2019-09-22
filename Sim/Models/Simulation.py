import pandas as pd
import yaml
import datetime
import os

class Simulation:
    
    kph2ms = 1/3.6
    
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
        
        self.data.insert(0, 'simName', [self.settings['meta']['name']]*len(self.data), True)
        
        self.splitStints()
    
    def getRoute(self):
        self.data = pd.read_csv(self.settings['route']['routeFile'])
            
        # Convert the locations list to a more user friendly dict
        locations = self.settings['route']['locations']
        self.locations = {}
        for iLocation in range(0, len(locations)):
            thisLocation = locations[iLocation]
            self.locations[thisLocation['name']] = {}
            self.locations[thisLocation['name']]['distance'] = thisLocation['distance']
            
    def splitStints(self):
        # Set the stints
        self.stints = self.settings['route']['stints']
        self.NStints = len(self.stints)
        
        for iStint in range(0, self.NStints):
            stintNumber = iStint+1
            
            print('Stint #{}'.format(stintNumber))
            
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
            
            # Set the distance between mesh locationss
            self.stints[iStint]['data'].at[0, 'd_distance'] = 0
            for i in range(1, len(self.stints[iStint]['data'])):
                self.stints[iStint]['data'].loc[i, 'd_distance'] = self.stints[iStint]['data'].loc[i, 'distance'] - self.stints[iStint]['data'].loc[i-1, 'distance']
            
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
            self.stints[iStint]['data'].insert(len(self.stints[iStint]['data'].columns), 'speed', self.stints[iStint]['averageSpeed'])
            self.stints[iStint]['data'].insert(len(self.stints[iStint]['data'].columns), 'speedms', self.stints[iStint]['averageSpeed']*Simulation.kph2ms)
            
            # Initialise the stint time
            self.stints[iStint] = self.calculateTime(self.stints[iStint])
        
    def calculateTime(self, stint):
        # Calculates the time using the car speed as input
        
        stint['data'].insert(len(stint['data'].columns), 'time', stint['startTime'])
        stint['data'].insert(len(stint['data'].columns), 'day', stint['startDay'])
        stint['data'].insert(len(stint['data'].columns), 'time_unix', stint['startTime'].timestamp())
        stint['data'].insert(len(stint['data'].columns), 'd_time', 0)
        stint['data'].insert(len(stint['data'].columns), 'd_timeDriving', 0)
        
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
                    print('Current: {}'.format(stint['data'].at[i-1, 'time'] + d_time))
                    print('End of Day: {}'.format(self.settings['time']['days'][stint['data'].day[i]-1]['end']))
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
                
        return stint
    
    def calculateAero(self, stint):
        CdA = self.settings['aero']['CdA']
        rho = 1.225
        
        stint['data'].insert(len(stint['data'].columns), 'aero__dragForce', CdA*0.5*rho*(stint['data'].speed*self.kph2ms)**2)
        stint['data'].insert(len(stint['data'].columns), 'aero__dragPower', stint['data'].aero__dragForce*stint['data'].speedms)
        stint['data'].insert(len(stint['data'].columns), 'aero__d_dragEnergy', 0)
        stint['data'].insert(len(stint['data'].columns), 'aero__dragEnergy', 0)
        
        for i in range(1, len(stint['data'])):
            averageDragPower = 0.5*stint['data'].aero__dragPower[i] + 0.5*stint['data'].aero__dragPower[i-1]
            stint['data'].at[i, 'aero__d_dragEnergy'] = averageDragPower*stint['data'].d_timeDriving[i].seconds
            stint['data'].at[i, 'aero__dragEnergy'] = stint['data'].at[i-1, 'aero__dragEnergy'] + stint['data'].at[i, 'aero__d_dragEnergy']
        
        return stint

    def initWeather(self):
        self.weather = {}
        
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