
#from astral import Location as AstralLocation

from Models.Simulation import Simulation


simulationSettingsPath = r'C:\Users\tom_m\Tommy Li\Github\RaceSim\Cases\20190812_Baseline.yml'

sim = Simulation(simulationSettingsPath)

for iStint in range(0,sim.NStints):
    bChangesMade = True
    iteration = 0
    
    while bChangesMade & (iteration < sim.settings['simulation']['iterLimit']):
        stint = sim.stints[iStint]
        iteration += 1
        
        # Run the models
        sim.calculateTime(stint)
        sim.runModels(stint)
        
        # Calculate the sensitivity to speed at each mesh point
        sim.calculateSensitivities(stint)
        
        # Adjust the speed of each mesh point
        # Increase speed at 'cheap' locations, decrease speed at 'expensive' locations
        # Increase speed at 'cheap' locations until arrival time constraint is met
        changesMade = sim.adjustSpeed(stint)
        bChangesMade = len(changesMade) > 0
        
        print('Stint #{} | iter {} | sensDelta: {:.6f} | arrivalDelta: {:4.2f} | changeMade: {}'.format(stint['nStint'], iteration, stint['data'].sens_powerPerKphDeltaToMin_gated.max(), stint['arrivalTimeDelta'], changesMade))


sim.combineStints()

#print(sim.settingsPath)
#print(sim.data)

sim.writeOutput()