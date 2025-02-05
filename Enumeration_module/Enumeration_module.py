import os
import numpy as np
import csv
import time
# from utils import Finding_Indicator_mat_n
# from utils_CSV import Finding_Indicator_mat
# from numba import prange
# from utils import Polytope_formation
# from utils_CSV import Polytope_formation
# from utils import saving_results
# from plot_res import plot_hyperplanes_and_vertices
# from plot_res import plot_polytope
# from plot_res import plot_polytope_2D
# from utils import Enumerator_rapid
from .utils_Enumeration import *
from .utils_Enumeration import Enumerator_rapid
from .utils_Enumeration import finding_deep_hype
from .utils_CSV import Enumerator
# from Preprocessing import preprocessing
# from Refinement_process import Refinement
import torch
# import multiprocessing
# from memory_profiler import profile

def enumeration_function(NN_file,name,TH,mode,parallel): 
    model = torch.jit.load(NN_file)
    #knowing number of neurons in each layer
    cntr=0
    params=[]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, param in model.named_parameters():
        with torch.no_grad():
            if device.type=='cuda':
                param=param.cpu()
                param=param.numpy()
                params.append(param)
            else:
                params.append(param.numpy())
        cntr=cntr+1
    num_hidden_layers = ((cntr-4)/2)+1
    print(num_hidden_layers)
            

    hyperplanes=[]
    b=[]
    W=[]
    c=[]
    nn=[]
    for i in range(len(params)-2):
            if i%2==0:
                hyperplanes.append(params[i])
                nn.append(np.shape(params[i])[0])
            else:
                b.append(params[i])
    W=params[-2]
    c=params[-1]
    shape=(1_000_000,6)
    D=[]
    # x=np.linspace(-2.0, 2.0, 200)
    # y=np.linspace(-2.0, 2.0, 200)
    # z=np.linspace(-2.0, 2.0, 200)
    # X,Y,Z=np.meshgrid(x,y,z)
    # X_train=np.vstack((X.flatten(),Y.flatten(),Z.flatten())).T

    # # for j in range(30):
    # #     X_train=np.random.uniform(-2, 2, size=shape)
    # #     # X_train=torch.tensor(X_train).float()
    # for i in X_train:
    #     val1=np.maximum(hyperplanes[0]@i+b[0],0)
    #     D1=np.sign(val1)
    #     val2=hyperplanes[1]@val1+b[1]
    #     D2=np.sign(val2)
    #     nD=np.hstack((D1,D2))
    #     D.append(nD)
    # D=list(np.unique(np.array(D),axis=0))
    n_h,n=np.shape(hyperplanes[0])
    original_polytope_test=np.array([generate_hypercube_vertices(n,TH,-TH)])
    cwd=os.getcwd()
    print(cwd)
    if mode=="Low_Ram":
        csv_file=cwd+'\Results'+'\Enumerate_poly_'+name+'.csv'
        with open (csv_file,'w',newline='') as f:
            wtr = csv.writer(f)
            wtr.writerows(original_polytope_test)
    border_hyperplane=np.vstack((np.eye(n),-np.eye(n)))
    border_bias=[-TH]*np.shape(border_hyperplane)[0]
    status=True
    enumeration_time=0
    D=np.array([1]*n_h)
    # enumerate_poly = []
    Enum=[]
    enumerate_poly=list(original_polytope_test)
    # start_process=time.time()
    st_enum=time.time()
    for i in range(int(num_hidden_layers)):
        for j in range(len(enumerate_poly)):
            # print("Layer=",i,"Cell=/Number of cells=",j,"/",len(enumerate_poly))
            if mode=="Low_Ram":
                enumerate_poly,border_hyperplane,border_bias=Enumerator(hyperplanes[i],b[i],original_polytope_test,TH,[border_hyperplane],[border_bias],csv_file,D,i,enumerate_poly,hyperplanes)
            else:
                if i==0:
                    enumerate_poly_n=Enumerator_rapid(hyperplanes[i],b[i],original_polytope_test,TH,[border_hyperplane],[border_bias],parallel,D,i)
                    Enum.extend(enumerate_poly_n)
                else:
                    # mid_point=np.mean(enumerate_poly[j],axis=0)

                    hype1,bias1,border_hyperplane1,border_bias1=finding_deep_hype(hyperplanes,b,enumerate_poly[j],border_hyperplane,border_bias,i,n)
                    enumerate_poly_n=Enumerator_rapid(hype1,bias1,np.array([enumerate_poly[j]]),TH,[border_hyperplane1],[border_bias1],parallel,D,i)
                    Enum.extend(enumerate_poly_n)
        end_enum=time.time()
        enumerate_poly=[]
        enumerate_poly.extend(Enum)
        Enum=[]
    D=[]
    # for i in enumerate_poly:
    #     mid_point=np.mean(i,axis=0)
    #     val1=np.maximum(hyperplanes[0]@mid_point+b[0],0)
    #     D1=np.sign(val1)
    #     val2=np.maximum(hyperplanes[1]@val1+b[1],0)
    #     D2=np.sign(val2)
    #     val2=np.maximum(hyperplanes[2]@val2+b[2],0)
    #     D3=np.sign(val2)
    #     nD=np.hstack((D1,D2,D3))
    #     D.append(nD)
    
    enumeration_time=enumeration_time+(end_enum-st_enum)

    end_process=time.time()
    print("Accumulative enumeration time=\n",enumeration_time)
    print("Number of hyperplanes:\n",n_h)
    print("Number of cells:\n",len(enumerate_poly))        
    # plot_polytope_2D(NN_file,TH)
    # plot_polytope(enumerate_poly,"test")   




def generate_hypercube_vertices(dimensions, lower_bound, upper_bound):
    if dimensions == 0:
        return [[]]

    vertices = []

    for vertex in generate_hypercube_vertices(dimensions - 1, lower_bound, upper_bound):
        for value in [upper_bound, lower_bound]:
            vertices.append([value] + vertex)

    return vertices




