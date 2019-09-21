import pandas as pd
import yaml

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
    
    def getRoute(self):
        self.data = pd.read_csv(self.settings['route']['routeFile'])
        
        

    def initWeather(self):
        self.weather = {}