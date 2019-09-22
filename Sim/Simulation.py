import pandas as pd
import yaml
import os

class Simulation:
    
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
        
        self.setInitialCondition()
    
    def getRoute(self):
        self.data = pd.read_csv(self.settings['route']['routeFile'])
        
        # Set the distance between mesh locations
        self.data.loc[0, 'dDistance'] = 0
        for i in range(1, len(self.data)):
            self.data.loc[i, 'dDistance'] = self.data.loc[i, 'distance'] - self.data.loc[i-1, 'distance']
            
        # Convert the locations list to a more user friendly dict
        locations = self.settings['route']['locations']
        self.locations = {}
        for iLocation in range(0, len(locations)):
            thisLocation = locations[iLocation]
            self.locations[thisLocation['name']] = {}
            self.locations[thisLocation['name']]['distance'] = thisLocation['distance']
            self.locations[thisLocation['name']]['isControlStop'] = thisLocation['name'] in self.settings['route']['controlStops']['locations']
    
    def splitStints(self):
        # Set the stints
        self.stints = self.settings['route']['stints']
        self.NStints = len(self.stints)
        
        for iStint in range(0, self.NStints):
            print('Stint #{}'.format(iStint))
            thisStint = self.stints[iStint]
            
            print(self.stints)
            
            startDistance = -1
            endDistance = -1
                
            startDistance = self.locations[thisStint['startLocation']]['distance']
            endDistance = self.locations[thisStint['endLocation']]['distance']
                
            print(startDistance)
            print(endDistance)
            
            # Find the section of the table that contains the current stint
            inStint = ((self.data.distance > startDistance) & (self.data.distance <= endDistance)).values.tolist()
            startIndex = inStint.index(True) - 1
            endIndex = len(inStint) - inStint[::-1].index(True) - 1
            
            # Get the data that belongs to this stint
            self.stints[iStint]['data'] = self.data.iloc[startIndex:endIndex+1, :]
            
            # Add some meta data
            self.stints[iStint]['startDistance'] = startDistance
            self.stints[iStint]['endDistance'] = endDistance
            self.stints[iStint]['meshPoints'] = endIndex+1-startIndex
            
            # Set the stint number (starting from 1)
            self.stints[iStint]['data'].insert(len(self.stints[iStint]['data'].columns), 'stintNumber', [iStint+1]*len(self.stints[iStint]['data']))
            
            
            
            
            
            
            
            print(self.stints[iStint])  
        
        

    def initWeather(self):
        self.weather = {}
        
    def setInitialCondition(self):
        print('Setting initial condition')
        
    def combineStints(self):
        for iStint in range(0, self.NStints):
            if iStint is 0:
                self.data = self.stints[iStint]['data']
            else:
                self.data = self.data.append(self.stints[iStint]['data'], ignore_index=True)
                
    def writeOutput(self):

        outputFolder = '{}\\..\\Cases'.format(os.getcwd())
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
            
        self.data.to_csv('{}\\{}.csv'.format(outputFolder, self.settings['meta']['name']))