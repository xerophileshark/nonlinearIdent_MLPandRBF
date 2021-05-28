# nonlinearIdent_MLPandRBF

In this simple project, we try to identify a model for a nonlinear dynamical system using RBF and MLP neural networks in MATLAB without using any toolbox.

## Dynamics of the system

The system dynamics evolve with the following differential equation

![dyn](https://user-images.githubusercontent.com/30368346/119955952-d183e580-bfb5-11eb-8308-29bc84090d8c.JPG)

where 

α=0.75 and β=1.5.

## Data
The following input is applied to the system in order to 

![inp](https://user-images.githubusercontent.com/30368346/119955959-d2b51280-bfb5-11eb-901b-c59fdd85a2ea.JPG),

161 learning data and 40 test data.

## MLP Network

The MLP neural network structure is as follows:

![MLP](https://user-images.githubusercontent.com/30368346/119957117-093f5d00-bfb7-11eb-86e8-e8f5e824b3cc.PNG)

where the activation function of hidden layer neurons are

![activation_hidden](https://user-images.githubusercontent.com/30368346/119958525-5d970c80-bfb8-11eb-917e-2a92683b2cd2.PNG)

and for output layer we have

![activation_output](https://user-images.githubusercontent.com/30368346/119958559-64258400-bfb8-11eb-87f2-bf01747bce6a.PNG)

Also:

![X_vect](https://user-images.githubusercontent.com/30368346/119958124-faa57580-bfb7-11eb-99d4-555999e91373.PNG)

![weight_vects](https://user-images.githubusercontent.com/30368346/119958567-65ef4780-bfb8-11eb-930d-3b36c134032f.PNG)

## RBF Network

The MLP neural network structure is as follows:

![RBF](https://user-images.githubusercontent.com/30368346/119957122-0a708a00-bfb7-11eb-8d67-6e1d649bcb11.PNG)

In this network, we employ a semi-supervised approach to train the weights.
