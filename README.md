## Code for paper titled: 
     `Physics Guided Design and Learning of Neural Networks for Predicting Drag Force on Particle Suspensions in Moving Fluids.`

## evaluation_and_visualization.py

## all_models.py

## utils.py

## notebooks/
- DNN.ipynb: This contains the feed-forward neural network model.
- DNN+Pres.ipynb: The DNN+ model with dragforce + pressure field prediction.
- DNN+Vel.ipynb:  The DNN+ model with dragforce + velocity field prediction.
- DNN-MT-Pres.ipynb: The DNN multi-task model with dragforce and pressure field prediction as two separate tasks. The architecture has a set
                      of shared layers and a few layers separate for each of the two tasks.
- DNN-MT-Vel.ipynb: The DNN multi-task model with dragforce and velocity field prediction as two separate tasks. The architecture has a set
                     of shared layers and a few layers separate for each of the two tasks.
- PhyDNN.ipynb: This is the proposed model with physics guided architecture and statistical priors incorporated via aggregate supervision.
- PhyDNN-PxTx.ipynb: Similar to PhyDNN.ipynb except that in this model, the drag force components (pressure and shear drag) are predicted only                              for the x-direction instead of the x,y,z directions.
