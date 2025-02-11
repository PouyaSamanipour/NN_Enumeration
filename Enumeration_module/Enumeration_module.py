import os
import numpy as np
import csv
import time
import itertools
from .plot_res import plot_polytope
# from utils import Enumerator_rapid
from .utils_Enumeration import *
from .utils_Enumeration import Enumerator_rapid
from .utils_Enumeration import finding_deep_hype
from .utils_CSV import Enumerator
import torch
import pickle

def enumeration_function(NN_file,name_file,TH,mode,parallel): 
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
    D=[]
    n_h,n=np.shape(hyperplanes[0])
    # original_polytope_test=np.array([generate_hypercube_vertices(n,TH,-TH)])
    original_polytope_test=np.array([generate_hypercube(TH)])
    cwd=os.getcwd()
    print(cwd)
    if mode=="Low_Ram":
        csv_file=cwd+'\Results'+'\Enumerate_poly_'+name+'.csv'
        with open (csv_file,'w',newline='') as f:
            wtr = csv.writer(f)
            wtr.writerows(original_polytope_test)
    border_hyperplane=np.vstack((np.eye(n),-np.eye(n)))
    border_bias=[]
    for f in range(n):
        border_bias.append(TH[f])
    border_bias.extend(border_bias)


# border_bias=[-TH]*np.shape(border_hyperplane)[0]
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
    enumeration_time=enumeration_time+(end_enum-st_enum)
    with open(name_file+"_polytope.pkl", "wb") as f:
        pickle.dump(enumerate_poly, f)


    end_process=time.time()
    print("Accumulative enumeration time=\n",enumeration_time)
    print("Number of hyperplanes:\n",n_h)
    print("Number of cells:\n",len(enumerate_poly))        
    # plot_polytope_2D(NN_file,TH)
    D_raw=Finding_cell_id(enumerate_poly,hyperplanes,b,num_hidden_layers)
    with open(name_file+"_cell_id.pkl", "wb") as f:
        pickle.dump(D_raw, f)
    if n==2:
        plot_polytope(enumerate_poly,"test")   



def generate_hypercube_vertices(dimensions, lower_bound, upper_bound):
    if dimensions == 0:
        return [[]]

    vertices = []

    for vertex in generate_hypercube_vertices(dimensions - 1, lower_bound, upper_bound):
        for value in [upper_bound, lower_bound]:
            vertices.append([value] + vertex)

    return vertices



def generate_hypercube(bounds):
    """
    Given a list of upper bounds for each dimension (e.g., [1,2,4]),
    generate all vertices of the corresponding hypercube.

    For each bound b in bounds, we consider two values: -b and b.
    Thus, if bounds is of length d, we get 2^d vertices.
    """
    # Create a list of tuples by taking the Cartesian product 
    # of (-b, b) for each bound b in the list.
    vertices = list(itertools.product(*[(-b, b) for b in bounds]))
    return vertices


def Finding_cell_id(enumerate_poly,hyperplanes,bias,num_hidden_layers):
    Mid_points=np.zeros((len(enumerate_poly),len(hyperplanes[0][0])))
    indx=0
    for i in enumerate_poly:
        sum=np.sum(i,axis=0)
        # sum=np.zeros(len(all_hyperplanes[0]))
        # for j in i:
            # sum=sum+j
        Mid_points[indx]=(sum/len(i))
        indx=indx+1
    # Mid_points=[np.mean(i,axis=0) for i in enumerate_poly]
    D_raw=[]
    points=np.array(Mid_points)
    for i in range(int(num_hidden_layers)):
        z=np.dot(hyperplanes[i],points.T)+bias[i].reshape(-1,1)
        points=np.maximum(0,z.T)
        z[z<0]=0
        z[z>0]=1
        D_raw.append(z)
    return D_raw