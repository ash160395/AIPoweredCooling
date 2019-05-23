# AIPoweredCooling

Problem to solve

In 2016, DeepMind AI minimized a big part of Google’s cost by reducing Google Data Centre Cooling Bill by 40% using their DQN AI model (Deep Q-Learning). In this case study, we will do something very similar. We will set up our own server environment, and we will build an AI that will be controlling the cooling/heating of the server so that it stays in an optimal range of temperatures while saving the maximum energy, therefore minimizing the costs. And just as DeepMind AI did, our goal will be to achieve at least 40% energy saving.

Environment to define
Before we define the states, actions and rewards, we need to explain how the server operates. We will do that in several steps. First, we will list all the environment parameters and variables by which the server is controlled. After that we will set the essential assumption of the problem, on which our AI will rely to provide a solution. Then we will specify how we will simulate the whole process. And eventually we will explain the overall functioning of the server, and how the AI plays its role.

Parameters:
• the average atmospheric temperature over a month
• the optimal range of temperatures of the server, which will be [18◦C,24◦C]
• the minimum temperature of the server below which it fails to operate, which will be −20◦C
• the maximum temperature of the server above which it fails to operate, which will be 80◦C
• the minimum number of users in the server, which will be 10
• the maximum number of users in the server, which will be 100
• the maximum number of users in the server that can go up or down per minute, which will be 5 • the minimum rate of data transmission in the server, which will be 20
• the maximum rate of data transmission in the server, which will be 300
• the maximum rate of data transmission that can go up or down per minute, which will be 10

Variables:
• the temperature of the server at any minute
• the number of users in the server at any minute
• the rate of data transmission at any minute
• the energy spent by the AI onto the server (to cool it down or heat it up) at any minute
• the energy spent by the server’s integrated cooling system that automatically brings the server’s tem- perature back to the optimal range whenever the server’s temperature goes outside this optimal range

All these parameters and variables will be part of our server environment and will influence the actions of the AI on the server.

The number of users and the rate of data transmission will be randomly fluctuating to simulate an actual server. This leads to randomness in the temperature and the AI has to understand how much cooling or heating power it has to transfer to the server so as to not deteriorate the server performance and at the same time, expend the least energy by optimizing its heat transfer.

Defining the states.

The input state st at time t is composed of the following three elements: 
1. The temperature of the server at time t.
2. The number of users in the server at time t.
3. The rate of data transmission in the server at time t.
Thus the input state will be an input vector of these three elements. Our future AI will take this vector as input, and will return the action to play at each time t.

Defining the actions.

The actions are simply the temperature changes that the AI can cause inside the server, in order to heat it up or cool it down. In order to make our actions discrete, we will consider 5 possible temperature changes from −3◦C to +3◦C, so that we end up with the 5 following possible actions that the AI can play to regulate the temperature of the server:

0 The AI cools down the server by 3◦C
1 The AI cools down the server by 1.5◦C
2 The AI does not transfer any heat to the server (no temperature change)
3 The AI heats up the server by 1.5◦C
4 The AI heats up the server by 3◦C

Defining the rewards.

The reward at iteration t is the energy spent on the server that the AI is saving with respect to the server’s integrated cooling system, that is, the difference between the energy that the unintelligent cooling system would spend if the AI was deactivated and the energy that the AI spends onto the server:
Reward = E tnoAI − EtAI

Rewardt = Energy saved by the AI between t and t + 1
= EtnoAI − EtAI
= |∆T noAI| − |∆T AI|


# AI Solution

Q-Learning into Deep Learning

Deep Q-Learning consists of combining Q-Learning to an Artificial Neural Network. Inputs are encoded vectors, each one defining a state of the environment. These inputs go into an Artificial Neural Network, where the output is the action to play. More precisely, let’s say the game has n possible actions, the output layer of the neural network is comprised of n output neurons, each one corresponding to the Q-values of each action played in the current state. Then the action played is the one associated with the output neuron that has the highest Q-value (argmax), or the one returned by the softmax method. In our case we will use argmax. And since Q-values are real numbers, that makes our neural network an ANN for Regression.
Hence, in each state st:
• the prediction is the Q-value Q(st,at) where at is chosen by argmax or softmax
• the target is rt + γmax(Q(st+1, a)) a
• the loss error between the prediction and the target is the squared of the Temporal Difference
Then this loss error is backpropagated into the network, and the weights are updated according to how much
  they contributed to the error.

Experience Replay

So far we have only considered transitions from one state st to the next state st+1. The problem with this is that st is most of the time very correlated with st+1. Therefore the network is not learning much. This could be way improved if, instead of considering only this one previous transition, we considered the last m transitions where m is a large number. This pack of the last m transitions is what is called the Experience Replay. Then from this Experience Replay we take some random batches of transitions to make our updates.

The Brain

The brain, or more precisely the deep neural network of our AI, will be a fully connected neural network, composed of two hidden layers, the first one having 64 neurons, and the second one having 32 neurons. And as a reminder, this neural network takes as inputs the states of the environment, and returns as outputs the Q-Values for each of the 5 actions. This artificial brain will be trained with a "Mean Squared Error" loss, and an Adam optimizer.

# Implementation

Step 1: Building the Environment
1. Step 1-1: Introducing and initializing all the parameters and variables of the environment.
2. Step 1-2: Making a method that updates the environment right after the AI plays an action.
3. Step 1-3: Making a method that resets the environment.
4. Step 1-4: Making a method that gives us at any time the current state, the last reward obtained, and whether the game is over.

Step 2: Building the Brain
1. Step 2-1: Building the input layer composed of the input states.
2. Step 2-2: Building the hidden layers with a chosen number of these layers and neurons inside each, fully connected to the input layer and between each other.
3. Step 2-3: Building the output layer, fully connected to the last hidden layer.
4. Step 2-4: Assembling the full architecture inside a model object.
5. Step 2-5: Compiling the model with a Mean-Squared Error loss function and a chosen optimizer.

Step 3: Implementing the Deep Reinforcement Learning Algorithm
1. Step 3-1: Introducing and initializing all the parameters and variables of the DQN model.
2. Step 3-2: Making a method that builds the memory in Experience Replay.
3. Step 3-3: Making a method that builds and returns two batches of 10 inputs and 10 targets

Step 4: Training the AI
1. Step 4-1: Building the environment by creating an object of the Environment class built in Step 1.
2. Step 4-2: Building the artificial brain by creating an object of the Brain class built in Step 2.
3. Step 4-3: Building the DQN model by creating an object of the DQN class built in Step 3.
4. Step 4-4: Choosing the training mode.
5. Step 4-5: Starting the training with a for loop over a chosen number of epochs.
6. Step 4-6: During each epoch we repeat the whole Deep Q-Learning process, while also doing some exploration 30% of the time.

Step 5: Testing the AI
1. Step 5-1: Building a new environment by creating an object of the Environment class built in Step 1.
2. Step 5-2: Loading the artificial brain with its pre-trained weights from the previous training.
3. Step 5-3: Choosing the inference mode.
4. Step 5-4: Starting the simulation.
5. Step 5-5: At each iteration (each minute), our AI only plays the action that results from its prediction, and no exploration or Deep Q-Learning training is happening whatsoever.
