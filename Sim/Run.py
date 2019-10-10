
#from astral import Location as AstralLocation

from Models.Simulation import Simulation
import time

t_start = time.time()

simulationSettingsPath = r'C:\Users\tom_m\Tommy Li\Github\RaceSim\Cases\20191010_Baseline.yml'

sim = Simulation(simulationSettingsPath)

iterations = []

for iStint in range(0,sim.NStints):
    stint = sim.stints[iStint]
    
    bChangesMade = True
    iteration = 0
    
    if stint['endDistance'] > stint['startDistance'] :
        if stint['arrivalTime'] > stint['startTime'] :
        
            while bChangesMade & (iteration < sim.settings['simulation']['iterLimit']):
                
                iteration += 1
                stint['interation'] = iteration
                
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
                
                print('Stint #{} | iter {} | sensDelta: {:.6f} | arrivalDelta: {:4.2f} | changeMade: {}\n'.format(stint['nStint'], iteration, stint['data'].sens_powerPerKphDeltaToMin_gated.max(), stint['arrivalTimeDelta'], changesMade))
                
            iterations.append(iteration)
            
        else:
            print('Stint #{} not simulated, start time is after arrival time'.format(stint['nStint']))
    else:
        print('Stint #{} not simulated, start distance is after end distance'.format(stint['nStsint']))

sim.combineStints()

#print(sim.settingsPath)
#print(sim.data)

sim.writeOutput()

t_end = time.time()

print('Simulation completed in {:4.2f} minutes'.format((t_end-t_start)/60))

for iStint in range(0,sim.NStints):
    stint = sim.stints[iStint]
    
    print('Stint #{}: {} iterations'.format(stint['nStint'], iterations[iStint]))
    
