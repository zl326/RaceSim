
#from astral import Location as AstralLocation

from Simulation import Simulation
from Models import AeroModel



simulationSettingsPath = r'C:\Users\tom_m\Tommy Li\Github\RaceSim\Cases\20190812_Baseline.yml'

sim = Simulation(simulationSettingsPath)
aeroModel = AeroModel.AeroModel(2)

print(sim.settingsPath)
print(sim.data)
print(aeroModel.settings)