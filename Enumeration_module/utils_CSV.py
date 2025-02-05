import numba as nb
from numba.typed import List
from numba import njit
import numpy as np
from scipy.spatial import Delaunay
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import time
import os
from scipy.optimize import linprog
import csv
from numba import prange
import ast
import pandas as pd
import pickle
# from memory_profiler import profile
def Polytope_formation(original_polytope, boundary_hyperplane, hyperplanes, b, hyperplane_val,Th):
    # Initialize empty lists to store the two polytopes
    poly1 = []
    poly2 = []

    # Extract hyperplanes and bias for the first polytope
    E1 = hyperplanes
    e1 = b

    # Initialize a list to store intersection points
    intersection_points = []

    # Iterate through the boundary hyperplanes
    for i in boundary_hyperplane:
        # Extract the coefficients for the second hyperplane
        E2 = i[0:-1]
        e2 = i[-1]

        # Check if the two hyperplanes are not parallel
        if (not np.all(E1 == -E2)) and (not np.all(E1 == E2)):
            # Calculate the intersection point of the two hyperplanes
            intersection = np.linalg.solve(np.vstack((E1, E2)), np.array([-e1, -e2]))

            # Check if the intersection point is within a certain norm threshold (0.9001)
            if np.linalg.norm(intersection, ord=np.inf) <= Th+0.00001:
            # if Convex_combination(original_polytope,intersection):
                # If the intersection point is valid, append it to the list
                intersection_points.append(intersection.tolist())
            else:
                print("Something is wrong")  # Debug message if the intersection point is not valid

    # Extract points from the original polytope based on hyperplane values
    poly1.extend((original_polytope[hyperplane_val >= -1e-10]).tolist())
    poly2.extend((original_polytope[hyperplane_val <= 1e-10]).tolist())

    # Add intersection points to both polytopes
    poly1.extend(intersection_points)
    poly2.extend(intersection_points)

    # Return the two polytopes
    return [poly1, poly2]

def Polytope_formation_hd(original_polytope, boundary_hyperplane, bias, hyperplane_val,Th):
    # Initialize empty lists to store the two polytopes
    poly1 = []
    poly2 = []
    hyp1=[]
    hyp2=[]
    bias1=[]
    bias2=[]
    # Initialize a list to store intersection points
    intersection_points = []

    # Iterate through the boundary hyperplanes
    for j, i in  enumerate(boundary_hyperplane):
        try:
            intersection = np.linalg.solve(i, -np.array(bias[j]))

            # if Convex_combination_nb(original_polytope,intersection):
            if np.linalg.norm(intersection, ord=np.inf) <= Th+0.00001:
                # If the intersection point is valid, append it to the list
                intersection_points.append(intersection)
            # elif np.linalg.norm(intersection, ord=np.inf) <= Th+0.00001:
            else:
                print("check solution,the solution is not in the convex set ")
            #     intersection_points.append(intersection.tolist())  
        except np.linalg.LinAlgError:
            print("no solution is found")
    if len(intersection_points)<len(original_polytope[0])-1:
        raise Warning("Number of intersection points must be at least n-1")
    # Extract points from the original polytope based on hyperplane values
    poly1=np.vstack((original_polytope[hyperplane_val >= -1e-13],intersection_points))
    poly2=np.vstack((original_polytope[hyperplane_val <= -1e-13],intersection_points))
    # poly1.append((original_polytope[hyperplane_val >= -1e-13]))
    # poly2.append((original_polytope[hyperplane_val <= 1e-13]))

    # # Add intersection points to both polytopes
    # poly1.(np.array(intersection_points))
    # poly2.extend(np.array(intersection_points))

    if len(poly1)<len(original_polytope[0])+1:
        print("problem, number of vertices in poly1")
    if len(poly2)<len(original_polytope[0])+1:
        print("problem, number of vertices in poly2")
    return [poly1.tolist(), poly2.tolist()]


# @profile
def Enumerator(hyperplanes, b, original_polytope_test,TH,boundary_hyperplanes,border_bias,csv_file):
    # Initialize a list to store the enumerated polytopes
    enumerate_poly = []
    # Iterate through the hyperplanes
    for i in range(len(hyperplanes)):
        print("Hyperplane:",i)
        intact_poly=[]
        time1=time.time()
        n=len(hyperplanes[i])
        poly_dumy = []
        hyperplane_val=[]
        # specific_rows=np.arange(id_var[0],id_var[1])
        # data= pd.read_csv(csv_file,header=None, skiprows = lambda x: x not in specific_rows)
        # # data=pd.read_csv(csv_file,header=None)
        # # for indx,row in data.iterrows():
        # #     print(row)
        with open(csv_file, mode ='r')as file:
            csvFile = csv.reader(file)
            for line in csvFile:
                enumerate_poly_j=enumerate_poly_j_formation(line)
                # enumerate_poly_j=[]
                # # s=tuple(line.split())
                # for i in range(len(line)):
                #     float_values = [float(val) for val in line[i][1:-1].split(',')]
                #     enumerate_poly_j.append(float_values)
                # enumerate_poly_j=np.array(enumerate_poly_j)
                # String_list=list(line)
                # enumerate_poly_j = [ast.literal_eval(string) for string in String_list]
                # enumerate_poly_j=np.array(enumerate_poly_j)

        # for k in enumerate_poly:
            # dum=np.dot(k,hyperplanes[i].T) + b[i]
                hyperplane_val_j=np.dot(enumerate_poly_j,hyperplanes[i].T) + b[i]
        # hyperplane_val = [np.dot(k,hyperplanes[i].T) + b[i] for k in enumerate_poly]
        # for k in hyperplane_val:
                if  np.min(hyperplane_val_j)<-1e-6 and np.max(hyperplane_val_j)>1e-6:
                    sgn_var_j=(np.max(hyperplane_val_j) * np.min(hyperplane_val_j))
                else:
                    sgn_var_j=np.max(hyperplane_val_j) * 0
        # Iterate through the enumerated polytopes
        # for j in range(len(enumerate_poly)):
                list_boundary=[]
                list_bias=[]
                if sgn_var_j < -1e-10:
                    if n==2:
                        valid_side=[]
                        # If there is a sign variation, calculate the boundary hyperplanes
                        boundary_hyperplane = ConvexHull(enumerate_poly_j).equations
                        side = []
                        sides1,hyp_f=finding_side(boundary_hyperplanes[0],enumerate_poly_j,border_bias[0])
                        sides = ConvexHull(enumerate_poly_j).simplices
                        # Iterate through the simplices of the polytope
                        for m in range(len(sides)):
                            if (hyperplane_val_j[sides[m][0]]) * (hyperplane_val_j[sides[m][1]]) < 0:
                                # If there's a change in sign along the simplex, consider it as a side
                                side.append(boundary_hyperplane[m])
                                # test_side.append(sides[m])
                        for m in range(len(sides1)):
                            if (np.max((hyperplane_val_j[sides1[m]]))>1e-10) and (np.min(hyperplane_val_j[sides1[m]]) < -1e-10):
                                valid_side.append(hyp_f[m])
                        # Calculate the new polytopes using Polytope_formation
                        original_polytope_test = Polytope_formation(enumerate_poly_j, side, hyperplanes[i], b[i], hyperplane_val_j,TH)
                        ########################
                        # Extend the temporary polytope list with the new polytopes
                        poly_dumy.extend(original_polytope_test)

                        # Keep track of the unwanted polytopes
                        # unwanted_polytop.append(enumerate_poly[j])
                    else:
                        valid_side=[]
                        # sides1 = ConvexHull(enumerate_poly[j]).simplices
                        sides,hyp_f=finding_side(np.array(boundary_hyperplanes[0]),enumerate_poly_j,np.array(border_bias[0]))
                        list_boundary,list_bias=finding_valid_side(sides,np.array(hyperplane_val_j),n,hyp_f,hyperplanes[i],b[i])
                        if len(list_boundary)<n-1:
                            raise Warning("check: Number of valid sides is less than $n-1$")
                        original_polytope_test=Polytope_formation_hd(enumerate_poly_j, list_boundary, list_bias, np.array(hyperplane_val_j),TH)
                        # Extend the temporary polytope list with the new polytopes
                        poly_dumy.extend(original_polytope_test)
                        # Keep track of the unwanted polytopes
                        # unwanted_polytop.append(enumerate_poly[j])
                else:
                    intact_poly.append(enumerate_poly_j.tolist())
        # all_hyp=np.vstack((boundary_hyperplanes[0],hyperplanes[i])).tolist()
        # all_bias=border_bias[0]+[b[i]]
        intact_poly.extend(poly_dumy)
        # with open(csv_file, 'wb') as f:
        #     pickle.dump(intact_poly, f)
        with open (csv_file,'w',newline='') as f:
            wtr = csv.writer(f)
            wtr.writerows(intact_poly)
        del intact_poly
        # with open('piclist.pkl', 'rb') as f:
        #     loaded=pickle.load(f)
        boundary_hyperplanes[0]=np.vstack((boundary_hyperplanes[0],hyperplanes[i])).tolist()
        border_bias[0]=border_bias[0]+[b[i]]
        # Extend the enumerate_poly list with the temporary polytopes
        # enumerate_poly.extend(poly_dumy)
        # enumerate_poly=removing_unwanted_poly(enumerate_poly,sgn_var)
        # enumerate_poly = [enumerate_poly.remove(e) for e in unwanted_polytop if e in enumerate_poly]
        
    # Return the list of enumerated polytopes
    return enumerate_poly_j,boundary_hyperplanes[0],border_bias[0]

 
def checking_sloution(slack,eps):
    if np.max(slack)>=eps:
        status=True
        print("refinement is required")
    else:
        status=False
    return status

# def Finding_Indicator_mat(enumerate_poly,all_hyperplanes,all_bias):
#     Mid_points=[np.mean(i,axis=0) for i in enumerate_poly]
#     D_raw=(np.dot(all_hyperplanes,np.array(Mid_points).T)+all_bias).T
#     D_raw[D_raw>0]=1
#     D_raw[D_raw<0]=0
#     D=D_raw.tolist()
#     del D_raw
#     return D
def saving_results(W_v,all_hyperplanes,all_bias,c_v,name,eps1,eps2,n_r):
    cwd=os.getcwd()
    new_cwd=cwd+"\Results"+"\\"+ name
    cntr=0
    file_name=new_cwd+"_"+str(cntr)
    SYSDATA={
       "W_v":W_v,
       "H":all_hyperplanes,
       "b":all_bias,
       "c_v":[c_v], 
       "epsilon1":[eps1],
       "epsilon2":[eps2],
       "Number of region":[n_r]
    }
    ctr=1
    name_new=name+"_"+str(ctr)+".m"
    while os.path.exists(cwd+"/Results/"+name_new):
      ctr=ctr+1
      name_new=name+"_"+str(ctr)+".m"
    with open(cwd+"/Results/"+name_new, 'w') as f:
      for key in SYSDATA.keys():
        f.write("%s=%s\n"%(key,SYSDATA[key]))

    

# def intersection_higherdim(vertex,hyperplane,b):
#     A=np.vstack((np.dot(vertex,hyperplane),np.array([1,1])))
#     B=np.array([b,1])
#     res=linprog([0,0], A_eq=A, b_eq=B, bounds=[(0,1),(0,1)])
#     sol=res.x
#     intersection=sol[0]*vertex[0]+sol[1]*vertex[1]
#     return intersection

@njit
def finding_side(boundary_hyperplanes,enumerate_poly,border_bias):
    # side=List()
    # hyp_f=List()
    side=List()
    hyp_f=[]
    n=len(boundary_hyperplanes[0])
    # test=np.reshape(border_bias,(len(border_bias),1))
    # test=border_bias.reshape((len(border_bias),-1))
    dum=np.dot(boundary_hyperplanes,enumerate_poly.T)+border_bias.reshape((len(border_bias),-1))
    # dum=np.dot(boundary_hyperplanes,(np.array(enumerate_poly)).T)+test
    for j,i in enumerate(dum):
        res=[k for k,j in enumerate(i) if np.abs(j)<1e-10]
        if len(res)>=n:
            if res not in side:
                side.append((res))
                hyp_f.append((np.append(boundary_hyperplanes[j],border_bias[j])))
    return side,hyp_f
@njit
def check_sides(k,sides,hyperplane_val):
    stat=True
    # lists=[]
    # for i in test_side:
    #     lists.extend([i])
    # hyperplane_val=[0 for i in hyperplane_val if i<=1e-6 and i>=-1e-6]
    common_elements=np.array(list(finding_intersection_list(k,sides)))
    # common_elements = list(set.intersection(*map(set, test_side)))
    if len(common_elements)==2:
        if (np.max(hyperplane_val[common_elements])>=1e-10) and (np.min(hyperplane_val[common_elements])<=-1e-10):
            stat=True
        else:
            stat=False
    elif len(common_elements)>2:
        # stat=False
        raise Warning("the number of common elements between sides should be 2. Now it is:",len(common_elements))
    else:
        stat=False
    return stat

# def Convex_combination(points, x):
#     n_points = len(points)
#     n_dim = len(x)
#     c = np.zeros(n_points)
#     Aeq = np.array(points).T
#     beq = x
#     Aub=np.ones((1,n_points))
#     bub=[1.001]
#     x_bounds = [(-1e-10, 1.001)] * n_points
#     lp = linprog(c,A_ub=Aub,b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=x_bounds, method='highs')
#     if not(lp.success):
#         print("stop")
#     return lp.success

# def Convex_combination_nb(points, x):
#     n_points = len(points)
#     n_dim = len(x)
#     c = np.zeros(n_points)
#     A = np.vstack((np.array(points).T, np.ones(n_points)))
#     b = np.concatenate((x, [1]))
#     x_bounds = [(0, 1.001)] * n_points
#     lp = linprog(c, A_eq=A, b_eq=b)
#     if lp.success:
#         if np.sum(lp.x)>0.99 and np.sum(lp.x)<1.001:
#             if np.max(lp.x)>0.9999999:
#                 stat=False
#                 print("The solution is:",lp.x)
#             else:
#                 stat=True
#     else:
#         stat=False
#     return stat


# @njit
# def hyperplane_val_calc(enumerate_poly, hyperplane,b):
#     hyperplane_val=[]
#     for k in enumerate_poly:
#         dum=[]
#         for j in k:
#             dum.append(np.dot(np.array(list(j)),hyperplane.T)+b)
#         hyperplane_val.append(np.array(dum)) 
#     return hyperplane_val
# @njit
# def funct_help(x):
#     list=List()
#     for y in x:
#         list.append(List(y))
#     return list
def finding_valid_side(sides,hyperplane_val,n,hyp_f,hyperplanes,b):
    valid_side=[]
    for m in range(len(sides)):
        if (np.max((hyperplane_val[sides[m]]))>1e-10) and (np.min(hyperplane_val[sides[m]]) < -1e-10):
            valid_side.append(m)
    list_boundary,list_bias=check_valid_side(valid_side,sides,hyperplane_val,hyp_f,hyperplanes,b,n)
    return list_boundary,list_bias
# def forming_enumerate_poly(enumerate_poly,sgn_var,id,j,enumerate_poly_j):
#     cntr=0
#     list_new=[]
#     enumerate_tact=enumerate_poly[:len(id[id<=len(sgn_var)-1])]
#     enumerate_intact=enumerate_poly[len(id[id<=len(sgn_var)-1]):]
#     for i,j in enumerate(sgn_var):
#         if j>-1e-10:
#             list_new.append(enumerate_poly[cntr:len(id[id==i])])
#             cntr=cntr+len(id[id==i])
#         else:
#             id=id[id!=i]            
#             id[id>i]=id[id>i]-1
#     if len(list_new)>=1:
#         list_new.extend(enumerate_intact)
#         enumerate_poly=list_new
#     else:
#         enumerate_poly=enumerate_intact
#     return enumerate_poly,id
# @njit
# def finding_poly_index(enumerate_poly,j,id):
#     enumerate_j=[]
#     for k in range(len(id)):
#         if id[k]==j:
#             enumerate_j.append(enumerate_poly[k])
#     return enumerate_j
# @njit
# def finding_hyp_val_index(hyperplane_val,j,id):
#     # val_j=[]
#     # for i in range(len(id)):
#     #     if id[i]==j:
#     #         val_j.append(hyperplane_val[i])
#     val_j=hyperplane_val[id==j]
#     return val_j
# def finding_sgn_var_index(hyperplane_val_j):
#     sgn_var=[]
#     if  np.min(hyperplane_val_j)<-1e-6 and np.max(hyperplane_val_j)>1e-6:
#         sgn_var.append(np.max(hyperplane_val_j) * np.min(hyperplane_val_j))
#     else:
#         sgn_var.append(np.max(hyperplane_val_j) * 0)
#     return sgn_var

# def finding_valid_side(sides,hyperplane_val,n,hyp_f,hyperplanes,b):
#     valid_side=[]
#     list_boundary=[]
#     list_bias=[]
#     for m in range(len(sides)):
#         if (np.max((hyperplane_val[sides[m]]))>1e-10) and (np.min(hyperplane_val[sides[m]]) < -1e-10):
#             valid_side.append(m)
#     # list_boun_hyp=[i for i in range(len(valid_side))]
#     comb=combinations(valid_side, n-1)
#     for f in comb:
#         stat=check_sides(f,sides,hyperplane_val) # to check if the chosen sides have any common points
#         if stat:
#             list_hype_test=[hyp_f[l][0:-1] for l in f]
#             list_bias_test=[hyp_f[l][-1] for l in f]
#             list_hype_test.append(hyperplanes)
#             list_boundary.append(list_hype_test)
#             list_bias_test.append(b)
#             list_bias.append(list_bias_test)
#     return list_boundary,list_bias
def check_valid_side(valid_side,sides,hyperplane_val,hyp_f,hyperplanes,b,n):
    list_boundary=[]
    list_bias=[]
    comb=combinations(valid_side, n-1)
    # comb_n=np.array(list(comb))
    list_boundary,list_bias=check_combination(comb,sides,hyperplane_val,hyp_f,hyperplanes,b)
    # for f in comb:
    #     stat=check_sides(f,sides,hyperplane_val) # to check if the chosen sides have any common points
    #     if stat:
    #         list_hype_test=[hyp_f[l][0:-1] for l in f]
    #         list_bias_test=[hyp_f[l][-1] for l in f]
    #         list_hype_test.append(hyperplanes)
    #         list_boundary.append(list_hype_test)
    #         list_bias_test.append(b)
    #         list_bias.append(list_bias_test)
    return list_boundary,list_bias

def removing_unwanted_poly(enumerate_poly,sgn_var):
        list_new=[]
        enumerate_tact=enumerate_poly[:len(sgn_var)]
        enumerate_intact=enumerate_poly[len(sgn_var):]
        for x in range(len(sgn_var)):
            if sgn_var[x]>-1e-10:
                list_new.append(enumerate_tact[x])
        if len(list_new)>=1:
            list_new.extend(enumerate_intact)
            enumerate_poly=list_new
        else:
            enumerate_poly=enumerate_intact
        return enumerate_poly
# @njit
def check_combination(comb_n,sides,hyperplane_val,hyp_f,hyperplanes,b):
    list_boundary=[]
    list_bias=[]
    for f in comb_n:
        stat=check_sides(f,sides,hyperplane_val) # to check if the chosen sides have any common points
        if stat:
            list_hype_test=[hyp_f[l][0:-1] for l in f]
            list_bias_test=[hyp_f[l][-1] for l in f]
            list_hype_test.append(hyperplanes)
            list_boundary.append(list_hype_test)
            list_bias_test.append(b)
            list_bias.append(list_bias_test)
    return list_boundary,list_bias
@njit
def finding_intersection_list(k,sides):
    samp_list=[sides[l] for l in k]
    intersection_p=set(samp_list[0])
    for i in range(1,len(samp_list)):
        intersection_p=intersection_p.intersection(set(samp_list[i]))
    return intersection_p

# @njit
def enumerate_poly_j_formation(line):
    enumerate_poly_j=[]
                # s=tuple(line.split())
    for i in range(len(line)):
        float_values = [float(val) for val in line[i][1:-1].split(',')]
        enumerate_poly_j.append(float_values)
    enumerate_poly_j=np.array(enumerate_poly_j)
    return enumerate_poly_j
