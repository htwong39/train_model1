from __future__ import division

# import random as rd
import numpy as np
# import gym
import tensorflow as tf
# import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
# import scipy.misc
import os
import data_gen

from network import Qnetwork, experience_buffer

from world import room
from world import agent


# --------------------------------------------------------------------------------

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
      
# --------------------------------------------------------------------------------

batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 50000. #How many steps of training to reduce startE to endE.
num_episodes = 100000 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 10 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
#h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network
#rmax =  10000
tf.reset_default_graph()

# contextual objects/configuration

n_sensors = 4 # number of sensors
n_control = 4 # number of actuators
item_range = 5 # range for all data
livingroom = room(n_sensors,item_range)

#generate training data

train_set = np.array(data_gen.generateTrainingSet())

#print(train_set[np.random.randint(0,15),8:])

train_index = np.random.randint(0,15)

man = agent(n_control,train_set[train_index,8:])

# network configurations

I_size = 12
O_size = 40

# learning objects
mainQN = Qnetwork(I_size,O_size,n_control)
targetQN = Qnetwork(I_size,O_size,n_control)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables,tau)
myBuffer = experience_buffer()

# ----------------------------
# training starts here
# ----------------------------

#Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE)/anneling_steps

#create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

#Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:

    sess.run(init)

    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)

    # train for num_episodes
    for i in range(num_episodes):
        
        # create new buffer for experiences
        episodeBuffer = experience_buffer()

        # random sensor assignment 
        state = np.hstack([train_set[train_index,0:8], np.random.randint(2,3,n_sensors)])

        # no match
        success = False

        #Sum of all rewards 
        rewardsAll = 0
        
        # intra-episode counter
        z = 0 
        
        # initialize actuators
        action = np.zeros(n_control)

        # some set of random action + network prediction 
        while z < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            # intra-episode counter
            z += 1
            # episode counter
            total_steps += 1
            
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                # generate random values
                action = np.random.randint(0,item_range,n_control)
            else:
                # predict from network with previous sensor assignment
                state_in = np.hstack([state[0:8],state[8:]/5]) 
                action = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[state_in]})

            # perform action (= adjustment of actuation in livingroom)
            state_new , reward , success = man.action(livingroom,action)
            
            state_new = np.hstack([train_set[train_index,0:8] ,state_new ])

            # record the experience to our episode buffer.
            episodeBuffer.add(np.reshape(np.array([state,action,reward,state_new,success]),[1,5]))

            # only if we are in the actual training phase
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                
                # perform training step every (four) steps
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.

                    # train with sampled episodes' s1 parts (4th item in array) 

                    Q_in = np.vstack(trainBatch[:,3])

                    Q_in = np.hstack([Q_in[:,0:8],Q_in[:,8:]/5])

                    Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:Q_in})
                    Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:Q_in})

                    end_multiplier = -(trainBatch[:,4] - 1)

                    Q1 = np.reshape(Q1,[batch_size,n_control]) 
                    Q2 = np.reshape(Q2,[batch_size,n_control,5])
                    
                    doubleQ = np.zeros((batch_size,n_control))

                    for l in range(0,n_control):
                        temp = Q2[:,l,:]
                        doubleQ[:,l] = temp[range(batch_size),Q1[:,l]]

                    doubleQ = np.transpose(doubleQ)

                    targetQ = np.transpose(np.vstack(trainBatch[:,2])) + (y * doubleQ * end_multiplier)

                    targetQ_add = np.transpose(targetQ)

                    action_add = np.vstack(trainBatch[:,1])

                    update_in = np.vstack(trainBatch[:,0])

                    update_in = np.hstack([update_in[:,0:8],update_in[:,8:]/5])
                    
                    _ = sess.run([mainQN.updateModel], \
                       feed_dict={mainQN.scalarInput:update_in,mainQN.targetQ:targetQ_add, mainQN.actions:action_add})
                    
                    #Update the target network toward the primary network.
                    updateTarget(targetOps,sess)

            livingroom.reset()
            train_index = np.random.randint(0,15)
            man.reset(n_control,train_set[train_index,8:])
            rewardsAll += reward
            state = state_new

            if success:
                break
        
        myBuffer.add(episodeBuffer.buffer)
#         jList.append(z)
        rList.append(rewardsAll)
        
        if i % 1000 == 0:
            saver.save(sess,path+'/model-'+str(i)+'.cptk')
            print("Saved Model") 
        if len(rList) % 10 == 0:
            print(total_steps,np.mean(rList[-10:]), e)
        if (i>100 and i%1000 == 0):
            plt.show(block=False)
            rMat = np.resize(np.array(rList),[len(rList)//100,100])
            rMean = np.average(rMat,1)
            plt.plot(rMean)
            plt.pause(0.05)
    saver.save(sess,path+'/model-'+str(i)+'.cptk')
print("Percent of successful episodes: " + str(sum(rList)/num_episodes) + "%")

#plt.show(block=True)
rMat = np.resize(np.array(rList),[len(rList)//100,100])
rMean = np.average(rMat,1)
plt.plot(rMean)
plt.show()
#plt.pause()
