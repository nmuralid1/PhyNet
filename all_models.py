import torch
import torch.nn as nn
import pdb
import random
import numpy as np
import os

def seed_everything(seed=123):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

#Model 1 Feed-forward neural network.
#This model can be used both for DNN and DNN+.
class DNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth,device='cpu'):
        super(DNN, self).__init__()
        
        # depth
        self.depth = depth
        
        # deploy layers
        self.input_layer = torch.nn.Linear(input_size, hidden_size).to(device)
        self.hidden_layers = nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size).to(device) for _ in range(depth)]) 
        self.output_layer  = torch.nn.Linear(hidden_size, output_size).to(device)
        
        # setting activations 
        self.input_activation = torch.nn.Tanh()
        self.hidden_activation = torch.nn.ELU()
        self.output_activation = 'Linear'
    
    def forward(self, x):
        # input layer
        out = self.input_layer(x)
        out = self.input_activation(out)
        
        # hidden layers
        for h_layer in self.hidden_layers:
            out = h_layer(out)
            out = self.hidden_activation(out)

        # output layer
        out = self.output_layer(out)
        
        return out

#PhyDNN model.
class MTNNConv_PressureVelocity(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size,output_size_pressurefield,output_size_velocityfield,
                 output_size_pdrag,output_size_vdrag, depth=2,depth_aux=2,shared_depth=1,conv_filter_size=3,
                 device="cpu",numpresdragcomponent=1,numsheardragcomponent=1):
        """
            @param input_size: Input vector size.
            @param hidden_size: Hidden size.
            @param output_size: Output size.
            @param output_size_aux1: Output size of pressure prediction task.
            @param output_size_aux2: Output size of velocity prediction task.
            @param depth: Number of hidden layers in main task.
            @param depth_aux: Number of hidden layers of auxiliary task.
            @param shared_depth: Number of hidden layers for shared representation.
            @param numpresdragcomponent: The number of pressure drag components that are being modeled. Ex: Either 1,2 or 3.
            @param numsheardragcomponent: The number of shear drag components that are being modeled. Ex: Either 1,2 or 3.

            A shared multi-task architecture wherein initial few layers are shared. 
            There are 3 tasks with the main task being the drag force prediction task and 
            the auxiliary tasks being Pressure field and Velocity field prediction.
            
            We utilize the entire dataset for this task. Here, the output is masked in places wherein 
            the auxiliary tasks have no data (they are marked with place-holders as zero).
        """

        super(MTNNConv_PressureVelocity, self).__init__()

        #Drag Force Prediction Task
        self.depth = depth
        self.input_layer = torch.nn.Linear(input_size, hidden_size).to(device)

        #Auxiliary Task Hidden Layers
        self.pressurefield_hidden_layers=nn.ModuleList([torch.nn.Linear(hidden_size,hidden_size).to(device) 
                                                        for _ in range(depth)])
        
        self.velocityfield_hidden_layers=nn.ModuleList([torch.nn.Linear(hidden_size,hidden_size).to(device) 
                                                        for _ in range(depth)])

        self.hidden_layers = nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size).to(device) 
                                            for _ in range(depth)])
        
        self.hidden_size_conv = ((output_size_pressurefield - conv_filter_size)//1) + 1
        
        self.conv=torch.nn.Conv1d(in_channels=2,out_channels=2,kernel_size=conv_filter_size,padding_mode='circular')
        self.pool=torch.nn.MaxPool1d(kernel_size=min(2,self.hidden_size_conv),padding=0)

        
        """
            Initially, the final output_layer that predicts the drag force will just take in the two predicted components
            of drag. Each predicted component has 3 scalar values (one each in the x,y,z directions) for a total of 6 values
            that have to be predicted.    
        """
        self.output_layer = torch.nn.Linear(numpresdragcomponent+numsheardragcomponent,output_size).to(device)   #This uses only PX, TauX as inputs.


        
        #Initial set of Shared Layers.
        self.shared_layers = nn.ModuleList([torch.nn.Linear(hidden_size,hidden_size).to(device) 
                                            for _ in range(shared_depth)])
        
        #Output layers.
        self.output_layer_pressurefield  = torch.nn.Linear(hidden_size, output_size_pressurefield).to(device)
        self.output_layer_velocityfield = torch.nn.Linear(hidden_size,output_size_velocityfield).to(device)
        self.output_layer_pressuredrag = torch.nn.Linear(4,output_size_pdrag).to(device) #4 is the size of the output of the max-pooling layer.
        self.output_layer_velocitydrag = torch.nn.Linear(4,output_size_vdrag).to(device)

        
        # setting activations 
        self.input_activation = torch.nn.Tanh()
        self.hidden_activation = torch.nn.ELU()
        self.output_activation = 'Linear'


    def forward(self, x):
        ######## Main-task ########

        # Input
        out = self.input_layer(x)
        out = self.input_activation(out)

        #Shared Hidden Layers
        for shared_layer in self.shared_layers:
            out = shared_layer(out)
            out = self.hidden_activation(out)

        # Auxiliary Task 1 Start.
        pressure_field = self.output_layer_pressurefield(out)
        # Auxiliary Task End.
        
        # Auxiliary Task 2 Start.
        velocity_field = self.output_layer_velocityfield(out)
        # Auxiliary Task End.

        fields=torch.cat((pressure_field.unsqueeze(1),velocity_field.unsqueeze(1)),dim=1)
        
        #Auxiliary Task 3 Start. (Predict Pressure Component of Drag.)
        conv_out=self.conv(fields)
        conv_out = self.hidden_activation(conv_out)
        conv_out = self.pool(conv_out)        
        
        conv_pressure_field=conv_out[:,0,:]
        conv_velocity_field=conv_out[:,1,:]
        
        pressure_drag_component = self.output_layer_pressuredrag(conv_pressure_field)
        velocity_drag_component = self.output_layer_velocitydrag(conv_velocity_field)
       
        
        out = self.output_layer(torch.cat((pressure_drag_component,velocity_drag_component),dim=1))
        

        return out,pressure_field,velocity_field,pressure_drag_component,velocity_drag_component


#Model 2 Multi-task Feed-forward neural network. Here there is no separate AUX layer. The shared layer directly outptus auxiliary results.
class MTNN2(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size,output_size_aux, depth=2,depth_aux=2,shared_depth=1,device="cpu"):
        
        super(MTNN2, self).__init__()
        
        #Drag Force Prediction Task
        self.depth = depth
        self.input_layer = torch.nn.Linear(input_size, hidden_size).to(device)
        self.hidden_layers = nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size).to(device) for _ in range(depth)]) 
        self.output_layer  = torch.nn.Linear(hidden_size, output_size).to(device)
        
        #Shared Layer
        self.shared_layers = nn.ModuleList([torch.nn.Linear(hidden_size,hidden_size).to(device) for _ in range(shared_depth)])
        self.output_layer_aux  = torch.nn.Linear(hidden_size, output_size_aux).to(device)
        
        # setting activations 
        self.input_activation = torch.nn.Tanh()
        self.hidden_activation = torch.nn.ELU()
        self.output_activation = 'Linear'
        
    
    def forward(self, x):
        ######## Main-task ########
        
        # Input
        out = self.input_layer(x)
        out = self.input_activation(out)
        
        #Shared Hidden Layers
        for shared_layer in self.shared_layers:
            out = shared_layer(out)
            out = self.hidden_activation(out)

        # Auxiliary Task Start.
        out_aux = self.output_layer_aux(out)
        # Auxiliary Task End.

        #Main-task Hidden Layers
        for h_layer in self.hidden_layers:
            out = h_layer(out)
            out = self.hidden_activation(out)          

        out = self.output_layer(out)
        ######## Main-task End #####  

        return out,out_aux


#Model 2 Multi-task Feed-forward neural network.
#class MTNN(torch.nn.Module):
#    def __init__(self, input_size, hidden_size, output_size,output_size_aux, depth=2,depth_aux=2,shared_depth=1,device="cpu"):
#        
#        super(MTNN, self).__init__()
#        
#        #Drag Force Prediction Task
#        self.depth = depth
#        self.input_layer = torch.nn.Linear(input_size, hidden_size).to(device)
#        self.hidden_layers = nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size).to(device) for _ in range(depth)]) 
#        self.output_layer  = torch.nn.Linear(hidden_size, output_size).to(device)
#        
#        #Shared Layer
#        self.shared_layers = nn.ModuleList([torch.nn.Linear(hidden_size,hidden_size).to(device) for _ in range(shared_depth)])
#        
#        #Auxiliary Task
#        self.depth_aux = depth_aux
#        self.input_layer_aux = torch.nn.Linear(input_size, hidden_size).to(device)
#        self.hidden_layers_aux = nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size).to(device) for _ in range(depth_aux)]) 
#        self.output_layer_aux  = torch.nn.Linear(hidden_size, output_size_aux).to(device)
#        
#        # setting activations 
#        self.input_activation = torch.nn.Tanh()
#        self.hidden_activation = torch.nn.ELU()
#        self.output_activation = 'Linear'
#        
#    
#    def forward(self, x):
#        ######## Main-task ########
#        
#        # Input
#        out = self.input_layer(x)
#        out = self.input_activation(out)
#        
#        #Shared Hidden Layers
#        for shared_layer in self.shared_layers:
#            out = shared_layer(out)
#            out = self.hidden_activation(out)
#        
#        #Main-task Hidden Layers
#        for h_layer in self.hidden_layers:
#            out = h_layer(out)
#            out = self.hidden_activation(out)          
#
#        out = self.output_layer(out)
#        ######## Main-task End #####
#        
#        
#        ###### Auxiliary Task ######
#        out_aux = self.input_layer_aux(x)
#        out_aux = self.input_activation(out_aux)
#        
#        #Shared Hidden Layers
#        for shared_layer in self.shared_layers:
#            out_aux = shared_layer(out_aux)
#            out_aux = self.hidden_activation(out_aux)
#        
#        #Auxiliary-Task Hidden Layers
#        for h_layer_aux in self.hidden_layers_aux:
#            out_aux = h_layer_aux(out_aux)
#            out_aux = self.hidden_activation(out_aux)
#        
#        out_aux = self.output_layer_aux(out_aux)
#        #Auxiliary Task End #######
#
#        return out,out_aux

#MTNN Model Wherein Pressure Model Is Predicted first, the outputs of which are passed into the DNN model predicting drag force.
#class MTNN_Sequential(torch.nn.Module):
#    def __init__(self, input_size,input_size_aux, hidden_size, output_size,output_size_aux, depth=2,depth_aux=2,shared_depth=1,device="cpu"):
#        
#        super(MTNN_Sequential, self).__init__()
#        
#        #Drag Force Prediction Task
#        self.depth = depth
#        self.input_layer = torch.nn.Linear(input_size, hidden_size).to(device)
#        self.hidden_layers = nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size).to(device) for _ in range(depth)]) 
#        self.output_layer  = torch.nn.Linear(hidden_size, output_size).to(device)
#        
#        #Shared Layer
#        #self.shared_layers = [torch.nn.Linear(hidden_size,hidden_size).to(device) for _ in range(shared_depth)]
#        
#        #Auxiliary Task
#        self.depth_aux = depth_aux
#        self.input_layer_aux = torch.nn.Linear(input_size_aux, hidden_size).to(device)
#        self.hidden_layers_aux = nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size).to(device) for _ in range(depth_aux)])
#        self.output_layer_aux  = torch.nn.Linear(hidden_size, output_size_aux).to(device)
#        
#        # setting activations 
#        self.input_activation = torch.nn.Tanh()
#        self.hidden_activation = torch.nn.ELU()
#        self.output_activation = 'Linear'
#        
#    
#    def forward(self, x):
#        
#        ###### Auxiliary Task ######
#        out_aux = self.input_layer_aux(x)
#        out_aux = self.input_activation(out_aux)
#        
#        #Shared Hidden Layers
#        #for shared_layer in self.shared_layers:
#        #    out_aux = shared_layer(out_aux)
#        #    out_aux = self.hidden_activation(out_aux)
#        
#        #Auxiliary-Task Hidden Layers
#        for h_layer_aux in self.hidden_layers_aux:
#            out_aux = h_layer_aux(out_aux)
#            out_aux = self.hidden_activation(out_aux)
#        
#        out_aux = self.output_layer_aux(out_aux)
#        ###### Auxiliary Task End ######
#        
#        #Concatenate Predicted Pressure with Original Input To Pass Into Network.
#        inp=torch.cat((x,out_aux),dim=1) 
#
#        ######## Main-task ########
#        out = self.input_layer(inp)
#        out = self.input_activation(out)
#        
#        
#        #Main-task Hidden Layers
#        for h_layer in self.hidden_layers:
#            out = h_layer(out)
#            out = self.hidden_activation(out)          
#
#        out = self.output_layer(out)
#        ######## Main-task End #####
#         
#        return out,out_aux
#
