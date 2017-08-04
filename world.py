import numpy as np
import random as rd

class room():
    
    def __init__(self,number_of_item,i_range):
        self.number_of_item = number_of_item
        self.sensormax = np.zeros(number_of_item)
        self.sensormin = np.zeros(number_of_item)
        self.sensor = np.zeros(number_of_item) 
        self.reward = np.zeros(number_of_item)

        for i in range(0,number_of_item):
            self.sensormax[i] = i_range
            self.sensormin[i] = 0

    def change_sensor(self,sensor_number,change):        
        if( (self.sensor[sensor_number] + change < self.sensormax[sensor_number])
            and (self.sensor[sensor_number] + change > self.sensormin[sensor_number] ) ):
            self.sensor[sensor_number] = self.sensor[sensor_number] + change 
    
    def reset(self):
        self.sensor = np.random.randint( 0 ,5 ,self.number_of_item )
        self.reward = np.zeros(self.number_of_item)


class agent():
    
    def __init__(self,number_of_item,ideal):
        self.ideal = ideal
        self.t = np.zeros(number_of_item) + 0.5
        '''
        for i in range(0, number_of_item):
            self.ideal[i] = rd.randint(-3, 3)
            #print(self.ideal)
            self.t[i] = rd.randint(-1, 1)
        '''

    def reset(self,number_of_item,ideal):
        self.ideal = ideal
        self.t = np.zeros(number_of_item) + 0.5
        '''
        for i in range(0, number_of_item):
            self.ideal[i] = rd.randint(-3, 3)
            #print(self.ideal)
            self.t[i] = rd.randint(-1, 1)
        '''
    def action(self,room,a):

        # why this? subtract half of range as a normalization step?
        a = a - 2

        # loop thru all sensors
        for i in range(0, room.number_of_item):
            if( (room.sensor[i] + a[i] < room.sensormax[i] ) and (room.sensor[i] + a[i] > room.sensormin[i]) ):
                room.sensor[i] = room.sensor[i] + a[i]

            if (abs(room.sensor[i] - self.ideal[i]) > self.t[i]):
                room.reward[i] = room.reward[i] - 1
                d = False 

            else:
                room.reward[i] = room.reward[i] + 10
                d = True

        # problem: d is only taken from last loop iteration, shouldn't this be an aggregate of all comparisons?
        return room.sensor, room.reward , d
