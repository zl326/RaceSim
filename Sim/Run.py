
#from astral import Location as AstralLocation

from Models.Simulation import Simulation
from Models.AeroModel import AeroModel



simulationSettingsPath = r'C:\Users\tom_m\Tommy Li\Github\RaceSim\Cases\20190812_Baseline.yml'

sim = Simulation(simulationSettingsPath)
#aeroModel = AeroModel(2)

for iStint in range(0,sim.NStints):
    
    stint = sim.stints[iStint]
    
    print('Processing Stint #{}'.format(stint['nStint']))
    
    stint = sim.calculateAero(stint)
    
    print(stint)


sim.combineStints()

#print(sim.settingsPath)
#print(sim.data)
#print(aeroModel.settings)

sim.writeOutput()