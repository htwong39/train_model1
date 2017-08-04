
## train with 3 users, 4 actuators (light1, light2, light3, sound) and 5 scenarios (range is 0 - 5) 

import random
import numpy as np 

class User:
    def __init__(self, name):
        self.name = name
        self.scenarios = {}

    def addScenarioPreferences(self, scenario_name, preferences):
        self.scenarios[scenario_name] = preferences

    def generateTrainingSet(self, index):
        names = ['work', 'tv', 'cleaning', 'holidays', 'dinner']
        
        results = list(map(lambda i: int(i == index), [0, 1, 2, 3, 4])) 
        results += list(map(lambda item: random.gauss(item, 0), self.scenarios[names[index]]))
        
        return results
  
# setup
userA = User('userA')
userA.addScenarioPreferences('work', [5, 5, 0, 0])
userA.addScenarioPreferences('tv', [1, 1, 0, 0])
userA.addScenarioPreferences('cleaning', [5, 5, 5, 4])
userA.addScenarioPreferences('holidays', [0, 0, 0, 0])
userA.addScenarioPreferences('dinner', [0, 5, 1, 0])

userB = User('userB')
userB.addScenarioPreferences('work', [5, 5, 0, 1])
userB.addScenarioPreferences('tv', [1, 1, 0, 0])
userB.addScenarioPreferences('cleaning', [5, 5, 5, 5])
userB.addScenarioPreferences('holidays', [0, 1, 0, 0])
userB.addScenarioPreferences('dinner', [0, 4, 2, 0])

userC = User('userC')
userC.addScenarioPreferences('work', [5, 5, 0, 2])
userC.addScenarioPreferences('tv', [1, 0, 0, 0])
userC.addScenarioPreferences('cleaning', [5, 5, 5, 0])
userC.addScenarioPreferences('holidays', [0, 0, 1, 0])
userC.addScenarioPreferences('dinner', [1, 5, 1, 0])

# generate from all 3 users for all 5 scenarios, yields 15 samples
def generateTrainingSet():
    results = []
    
    # loop thru 'work', 'tv', 'cleaning', 'holidays', 'dinner'
    for index in range(0, 5):
        results.append([1, 0, 0] + userA.generateTrainingSet(index))
        results.append([0, 1, 0] + userB.generateTrainingSet(index))
        results.append([0, 0, 1] + userC.generateTrainingSet(index))
    return results     

#fullSet = np.array(generateTrainingSet())

#print(fullSet[0,:])

## train with:
#trSet = fullSet[:,0:8]
#print(trSet)

## generate rewards with:
#outputSet = fullSet[:,8:]
#print(fullSet.shape)
