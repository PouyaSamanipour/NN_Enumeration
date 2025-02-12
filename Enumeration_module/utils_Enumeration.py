import numba as nb
from numba.typed import List
from numba import njit
import numpy as np
# from scipy.spatial import Delaunay
from itertools import combinations,product
import time
import os
from scipy.optimize import linprog
import csv
from numpy.linalg import matrix_rank
from numpy.linalg import inv
from numba import prange
import torch
# import cdd
import math
def Polytope_formation(original_polytope, boundary_hyperplane, hyperplanes, b, hyperplane_val,Th):
    # Initialize empty lists to store the two polytopes
    poly1 = []
    poly2 = []

    # Extract hyperplanes and bias for the first polytope
    E1 = hyperplanes
    e1 = b
    Threshold=1e-12

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
            if  np.all(Th-intersection>-1e-05) and np.all(intersection+Th>-1e-05):
            # if np.linalg.norm(intersection-Th, ord=np.inf) <= 0.00001:
            # if Convex_combination(original_polytope,intersection):
                # If the intersection point is valid, append it to the list
                intersection_points.append(intersection.tolist())
            else:
                print("Something is wrong")  # Debug message if the intersection point is not valid

    # # Extract points from the original polytope based on hyperplane values
    # poly1.extend((original_polytope[hyperplane_val >= -1e-10]).tolist())
    # poly2.extend((original_polytope[hyperplane_val <= 1e-10]).tolist())

    # # Add intersection points to both polytopes
    # poly1.extend(intersection_points)
    # poly2.extend(intersection_points)
    if len(intersection_points)>0:
        poly1=np.vstack((original_polytope[hyperplane_val >= -Threshold],intersection_points))
        poly2=np.vstack((original_polytope[hyperplane_val <= Threshold],intersection_points))
    else:
        poly1=original_polytope[hyperplane_val >= -Threshold]
        poly2=original_polytope[hyperplane_val <= Threshold]

    if len(poly1)<len(original_polytope[0])+1:
        raise Warning("problem, number of vertices in poly1, You need to change the Threshold for this problem")
    if len(poly2)<len(original_polytope[0])+1:
        raise Warning("problem, number of vertices in poly2, You need to change the Threshold for this problem")
    return [poly1, poly2]    
    # poly1=np.vstack((original_polytope[hyperplane_val >= -1e-13],intersection_points))
    # poly2=np.vstack((original_polytope[hyperplane_val <= 1e-13],intersection_points))
    # Return the two polytopes
    return [poly1, poly2]
# @njit
def Polytope_formation_hd(original_polytope, hyperplane_val,Th,intersection_test):
    # Initialize empty lists to store the two polytopes
    # poly1 = []
    # poly2 = []
    Threshold=1e-12
    if len(intersection_test)>0:
            poly1 = []
            poly2 = []
            if len(intersection_test)<len(original_polytope[0])-1:
                raise Warning("Number of intersection points must be at least n-1")
            # Extract points from the original polytope based on hyperplane values
            poly1=np.vstack((original_polytope[hyperplane_val >= -Threshold],intersection_test))
            poly2=np.vstack((original_polytope[hyperplane_val <= Threshold],intersection_test))
            # poly1.append((original_polytope[hyperplane_val >= -1e-13]))
            # poly2.append((original_polytope[hyperplane_val <= 1e-13]))

            # # Add intersection points to both polytopes
            # poly1.(np.array(intersection_points))
            # poly2.extend(np.array(intersection_points))
    else:
        poly1=(original_polytope[hyperplane_val >= -Threshold])
        poly2=original_polytope[hyperplane_val <= Threshold]
    if len(poly1)<len(original_polytope[0])+1:
        raise Warning("problem, number of vertices in poly1")
    if len(poly2)<len(original_polytope[0])+1:
        raise Warning("problem, number of vertices in poly2")
    return [poly1, poly2]


def Enumerator_rapid(hyperplanes, b, original_polytope_test,TH,boundary_hyperplanes,border_bias,parallel,D,m):
    enumerate_poly=list(original_polytope_test)
 
    ##############################################################
        # Iterate through the hyperplanes
    for i in range(len(hyperplanes)):
        signs=[]
        # print("Neurons:",i)
        # if len(enumerate_poly)==0:
        #     # If it's the first iteration, initialize the enumerate_poly with a copy of the original polytope

        #     S=[-1]*len(boundary_hyperplanes[0])
        intact_poly=[]
        time1=time.time()
        n=len(enumerate_poly[0][0])
        poly_dumy = []
        hyperplane_val=[]
        sgn_var=[]
        for k in enumerate_poly:
            dum=np.dot(k,hyperplanes[i].T) + b[i]
            hyperplane_val.append(dum)
            if  np.min(dum)<-1e-5 and np.max(dum)>1e-5:
                sgn_var.append(np.max(dum) * np.min(dum))
                if sgn_var[-1]<-1e-9:
                    mid_point=np.mean(k,axis=0)
                    if np.all(mid_point==np.zeros((n,1))):
                        mid_point=mid_point+1e-10
                    sgn=np.dot(boundary_hyperplanes[0],mid_point)+border_bias[0]
                    signs.append(np.sign(sgn))
            else:
                sgn_var.append(np.max(dum) * 0)





















        # Iterate through the enumerated polytopes
        st100=time.time()
        for j in range(len(enumerate_poly)):

            list_boundary=[]
            list_bias=[]
            # if i==6 and j==13:
            #     print("check")
            if sgn_var[j] < -1e-9:
                if n==2:
                    valid_side=[]
                    # If there is a sign variation, calculate the boundary hyperplanes
                    # boundary_hyperplane = ConvexHull(enumerate_poly[j]).equations
                    side = []
                    # sides1,hyp_f=finding_side(np.array(boundary_hyperplanes[0]),enumerate_poly[j],np.array(border_bias[0]))
                    # sides = ConvexHull(enumerate_poly[j]).simplices
                    sides1,hyp_f=finding_side(np.array(boundary_hyperplanes[0]),enumerate_poly[j],np.array(border_bias[0]))
                    for m in range(len(sides1)):
                        if (hyperplane_val[j][sides1[m][0]]) * (hyperplane_val[j][sides1[m][1]]) < 0:
                            # If there's a change in sign along the simplex, consider it as a side
                            side.append(hyp_f[m])
                    # Iterate through the simplices of the polytope
                    # for m in range(len(sides)):
                    #     if (hyperplane_val[j][sides[m][0]]) * (hyperplane_val[j][sides[m][1]]) < 0:
                    #         # If there's a change in sign along the simplex, consider it as a side
                    #         side.append(boundary_hyperplane[m])
                    # Calculate the new polytopes using Polytope_formation
                    original_polytope_test = Polytope_formation(enumerate_poly[j], side, hyperplanes[i], b[i], hyperplane_val[j],TH)
                    ########################
                    # Extend the temporary polytope list with the new polytopes
                    poly_dumy.extend(original_polytope_test)
                    del original_polytope_test
                    # Keep track of the unwanted polytopes
                    # unwanted_polytop.append(enumerate_poly[j])
                else:
                    # intersection_n[index_list[i]:index_list[i+1]]
                    original_polytope_test=process_polytope(boundary_hyperplanes, enumerate_poly[j], border_bias, n, hyperplane_val[j], hyperplanes[i], b[i], TH, parallel)
                    # Extend the temporary polytope list with the new polytopes
                    poly_dumy.extend(original_polytope_test)
                    del original_polytope_test
                    # Keep track of the unwanted polytopes
                    # unwanted_polytop.append(enumerate_poly[j])
            else:
                intact_poly.append(enumerate_poly[j])
        end=time.time()
        # print("Duration:",end-st100)
        # all_hyp=np.vstack((boundary_hyperplanes[0],hyperplanes[i])).tolist()
        # all_bias=border_bias[0]+[b[i]]
        intact_poly.extend(poly_dumy)
        boundary_hyperplanes[0]=np.vstack((boundary_hyperplanes[0],hyperplanes[i])).tolist()
        border_bias[0]=border_bias[0]+[b[i]]
        enumerate_poly=intact_poly
        # enumerate_poly = [enumerate_poly.remove(e) for e in unwanted_polytop if e in enumerate_poly]
        
    # Return the list of enumerated polytopes
    return enumerate_poly

 
def checking_sloution(slack,eps):
    if np.max(slack)>=eps:
        status=True
        print("refinement is required")
    else:
        status=False
    return status
@njit
def Finding_Indicator_mat(enumerate_poly,all_hyperplanes,all_bias):
    Mid_points=np.zeros((len(enumerate_poly),len(all_hyperplanes[0])))
    indx=0
    for i in enumerate_poly:
        sum=np.sum(i,axis=0)
        # sum=np.zeros(len(all_hyperplanes[0]))
        # for j in i:
            # sum=sum+j
        Mid_points[indx]=(sum/len(i))
        indx=indx+1
    # Mid_points=[np.mean(i,axis=0) for i in enumerate_poly]
    D_raw=(np.dot(all_hyperplanes,Mid_points.T)+all_bias).T
    return D_raw
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

    
@njit
def finding_side(boundary_hyperplanes,enumerate_poly,border_bias):
    # side=list()
    # hyp_f=List()
    side=List()
    # side=[]
    hyp_f=[]
    n=len(boundary_hyperplanes[0])
    # test=np.reshape(border_bias,(len(border_bias),1))
    # test=border_bias.reshape((len(border_bias),-1))
    dum=np.dot(boundary_hyperplanes,enumerate_poly.T)+border_bias.reshape((len(border_bias),-1))
    # dum=np.dot(boundary_hyperplanes,(np.array(enumerate_poly)).T)+test
    for j,i in enumerate(dum):
        res=[k for k,l in enumerate(i) if np.abs(l)<1e-10]
        if len(res)>=n:
            if res not in side:
                side.append(((res)))
                hyp_f.append((np.append(boundary_hyperplanes[j],border_bias[j])))
                # vertices=(dum[j])[dum[j]<1e-10 and dum[j]>-1e-10]
    return side,hyp_f
@njit
def check_sides(k,sides,hyperplane_val):
    stat=True
    # print("k is:",k)
    # lists=[]
    # for i in test_side:
    #     lists.extend([i])
    # hyperplane_val=[0 for i in hyperplane_val if i<=1e-6 and i>=-1e-6]
    # print('k is :',k)
    # st=time.time()
    # common_elements=finding_common_elements(k,sides)
    # end=time.time()
    # st1=time.time()
    common_elements=np.array(list(finding_intersection_list(k,sides)))
    # end1=time.time()
    # print("New_func:",end-st)
    # print("Old_func",end1-st1)
    # common_elements = list(set.intersection(*map(set, test_side)))
    if len(common_elements)==2:
        if (np.max(hyperplane_val[common_elements])>=1e-10) and (np.min(hyperplane_val[common_elements])<=-1e-10):
            stat=True
            # print(common_elements)
        else:
            stat=False
    elif len(common_elements)>2:
        print("k in common elements:",k)
        print("common_elements:",common_elements)
        # stat=False
        raise Warning("the number of common elements between sides should be 2. Now it is:",len(common_elements))
    else:
        stat=False
    return stat

def finding_valid_side(sides,hyperplane_val,n,hyp_f,hyperplanes,b,TH,parallel,sign_m):
    valid_side=[]
    # restored_list = [list(filter(lambda x: x != -1, sublist)) for sublist in sides]
    for m in range(len(sides)):
        if (np.max((hyperplane_val[sides[m]]))>1e-10) and (np.min(hyperplane_val[sides[m]]) <-1e-10):
            valid_side.append(m)                   
    # st=time.time()
    # intersections_test=check_valid_side(valid_side,sides,hyperplane_val,hyp_f,hyperplanes,b,n,TH,parallel,sign_m)
    # print("Duraion:",time.time()-st)
    # st=time.time()
    # intersections_test=check_valid_side(valid_side,sides,hyperplane_val,hyp_f,hyperplanes,b,n,TH,parallel,sign_m)
    # if len(intersections_test1)!=len(intersections_test):
    #     print("check")
    # print("Duraion1:",time.time()-st)
    if len(valid_side)>0:
        intersections_test=check_valid_side(valid_side,sides,hyperplane_val,hyp_f,hyperplanes,b,n,TH,parallel,sign_m)
    else:
        intersections_test=[]
    return intersections_test

def check_valid_side(valid_side,sides,hyperplane_val,hyp_f,hyperplanes,b,n,TH,parallel,sign_m):
    lst_new,lst_dum=removing_unnecessary_comb(np.array(valid_side),sides,n,hyperplane_val)
    # chunks=[list(combinations(i,n-2)) for i in lst_new]
    # chunks=[]
    # for i in lst_new:
    #     lst_dum=np.array(list(combinations(i[1:],n-2)))
    #     lst_dum1=np.insert(lst_dum, 0,i[0], axis=1)
    #     chunks.extend(lst_dum1)
    
    
    ##################################################
        ####Multiprocessing#################
    # processes=6
    # chunk_size = len(chunks) // processes
    # chunks_n = [chunks[i : i + chunk_size] for i in range(0, len(chunks), chunk_size)]
    # pool = multiprocessing.Pool(processes=processes)
    # start_time = time.monotonic()
    # # n.append(chunks)
    # lst_n=[]
    # # lst_n.append((sides,hyperplane_val,np.array(hyp_f),hyperplanes,b))
    # st=time.time()
    # func = partial(new_test,list(sides),hyperplane_val,np.array(hyp_f),hyperplanes,b)
    # results =pool.map(func,chunks_n)
    # end=time.time()
    # print(f"time {end-st:0.2f}")        
    #comb_n=comb_with_excludes(valid_side, n-1, lst_test)
    # for i in comb_n:
    #     print(i)    
    # len1=len(list(comb_n))
    # ln=math.comb(len(lst_new[0][1:]),n-2)
    # comb=np.array(list(combinations(valid_side,n-1)))
    # list_boundary1,list_bias1=check_combination(comb,sides,hyperplane_val,np.array(hyp_f),hyperplanes,b)
    # print("Len_new:",len1)
    # print("ln:",ln)
    #######################################################################################    
    len_batch=500_000
    batch=0
    # if ln>len_batch:
    #     batch=int(ln/len_batch)
    # if ln % len_batch!=0:
    #     batch=batch+1
    cntr=0
    list1=[]
    intersection_test=[]
    # if ln >len_batch:
    #     print("check")
    # st=time.time()
    sum=0
    list_final=[]
    for j in range(len(lst_new)):
        batch=0
        ln=math.comb(len(lst_new[j][1:]),n-2)
        if ln>len_batch:
            batch=int(ln/len_batch)
        if ln % len_batch!=0:
            batch=batch+1
        cntr=0
        list1=np.array(list(combinations(lst_new[j][1:], n-2)))
        list1=np.insert(list1, 0,lst_new[j][0], axis=1)
        if len(lst_dum)>0:
            if len(list1)>5000:
                list1=tailoring_lists_parallel(list1,np.array(lst_dum))
                list1= list1[~np.all(list1 == 0, axis=1)]
            else:
                list1=tailoring_lists(list1,np.array(lst_dum))
        list_final.extend(list1)
        if parallel:    
            if len(list_final)>=len_batch or j==len(lst_new)-1:
                if len(list_final)>=10000:
                    st=time.time()
                    valid_combination=check_combination_parallel(TH,np.array(list_final),n,sides,hyperplane_val,np.array(hyp_f),hyperplanes,b,sign_m)                    
                    del list_final
                else:
                    valid_combination=check_combination(np.array(list_final),sides,hyperplane_val,np.array(hyp_f),hyperplanes,b,np.array(TH))
                    del list_final
                list_final=[]
                intersection_test.extend(valid_combination)
        else: 
            if len(list_final)>=len_batch or j==len(lst_new)-1:
                valid_combination=check_combination(np.array(list_final),sides,hyperplane_val,np.array(hyp_f),hyperplanes,b,np.array(TH))
                list_final=[]
                intersection_test.extend(valid_combination)


        # for i in range(batch):
        #     # comb=combinations(lst_new[i][1], n-1)
        #     if i<batch-1:
        #         list1=np.array(list(combinations(lst_new[j][1:], n-2))[cntr:cntr+len_batch])
        #     else:
        #         list1=np.array(list(combinations(lst_new[j][1:], n-2))[cntr:])
        #     # list1=np.array([j for indx,j in enumerate(comb) if cntr<=indx<cntr+len_batch])
        #     list1=np.insert(list1, 0,lst_new[j][0], axis=1)
        #     sum=len(list1)+sum
        #     cntr=cntr+len_batch
        #     # st=time.time()
        #     if parallel and len(list1)>1000:
        #         # st10=time.time()
        #         if len(lst_dum)>0:
        #             lst_n=tailoring_lists(list1,np.array(lst_dum))
        #             del list1 
        #             if len(lst_n)>0:
        #                 list11=np.array(lst_n)
        #                 valid_combination=check_combination_parallel(TH,list11,n,sides,hyperplane_val,hyp_f,hyperplanes,b,sign_m)
        #             else:
        #                 valid_combination=[]
        #             # st=time.time()
        #             # print("Dur_refined:",time.time()-st10)
        #         else:
        #             # st=time.time()
        #             if len(lst_dum)>0:
        #                 lst_n=tailoring_lists(list1,np.array(lst_dum))
        #                 del list1 
        #             if len(lst_n)>0:
        #                 list11=np.array(lst_n)
        #                 valid_combination=check_combination_parallel(TH,list11,n,sides,hyperplane_val,hyp_f,hyperplanes,b,sign_m)
        #             else:
        #                 valid_combination=[]
        #             del list1
        #             # print("Dur_parallel:",time.time()-st)
        #     else:
        #         valid_combination=check_combination(list1,sides,hyperplane_val,np.array(hyp_f),hyperplanes,b,TH)
        #         del list1
        #     # list_boundary.extend(list_boundary_pre)
        #     # list_bias.extend(list_bias_pre)
        #     intersection_test.extend(valid_combination)
    # end=time.time()
    # print("Duration:",end-st,"# combination:",math.comb(len(valid_side),n-2))
    # print("list1:",sum)
    return intersection_test

@njit
def finding_intersection_list(k,sides):
    # print(k)
    # restored_list=[list(filter(lambda x: x != -1, sublist)) for sublist in sides]
    samp_list=[sides[l] for l in k]
    intersection_p=set(samp_list[0])
    for i in range(1,len(samp_list)):
        intersection_p=intersection_p.intersection(set(samp_list[i]))
    return intersection_p


@njit
def check_combination(comb_n,sides,hyperplane_val,hyp_f,hyperplanes,b,TH):
    n=len(hyp_f[0])-1
    valid_comb=100*np.ones((1,n))
    list_sides=np.zeros((n,n))
    b_new=np.zeros((n,1))
    flag=0
    for f in comb_n:
        stat=check_sides(f,sides,hyperplane_val)
        if stat:
            list_sides[0:n-1]=(hyp_f[f])[:,0:-1]
            list_sides[n-1]=hyperplanes
            b_new[0:n-1]=(hyp_f[f])[:,n:n+1]
            b_new[n-1]=b
            dum=np.linalg.solve(list_sides,-b_new)
            if np.all(TH-dum[:,0]>-1e-05) and np.all(TH-dum[:,0]>-1e-05):
            # if np.linalg.norm(dum, ord=np.inf) <= TH+0.00001:
                valid_comb=np.vstack((valid_comb,dum.T))
    ret=valid_comb[1:]        
    return ret

@njit
def removing_unnecessary_comb(valid_side,sides,n,hyperplane_val):
    lst_test=[[valid_side[i]] for i in range(len(valid_side))]
    list_dum=[]
    for i in range(len(valid_side)-1):
        for j in range(i+1,len(valid_side)):
            dum=finding_intersection_list((valid_side[i],valid_side[j]),sides)
            if len(dum)>=n-1:
                val=hyperplane_val[np.array(list(dum))]
                if np.max(val)>1e-10 and np.min(val)<-1e-10: 
                    lst_test[i].extend([(valid_side[j])])
            else:
                list_dum.append([valid_side[i],valid_side[j]])
    lst_new=[i for i in lst_test if len(i)>=n-1]
    return lst_new,list_dum





























#############################Combination Parallel############3
def check_combination_parallel(TH,list1,n,sides,hyperplane_val,hyp_f,hyperplanes,b,sign_m):
    # list1_n=check_combination_test(list1,sides,hyperplane_val)
    valid_comb1=1000*np.ones((len(list1),n))
    valid_combination=check_combination_p(np.array(list1),sides,hyperplane_val,np.array(hyp_f),hyperplanes,b,valid_comb1,sign_m)
    valid_combination = valid_combination[~np.all(valid_combination == 1000, axis=1)]
    # valid_combination1=check_combination_p1(np.array(list1),sides,hyperplane_val,np.array(hyp_f),hyperplanes,b,TH,valid_comb1,sign_m)
    # valid_combination1 = valid_combination[~np.all(valid_combination == 10*TH, axis=1)]
    del valid_comb1

    
    return list(valid_combination)

@njit(parallel=True,fastmath=True)
def check_combination_p(comb_n,sides,hyperplane_val,hyp_f,hyperplanes,b,valid_comb,sign_m):    
    n=len(hyp_f[0])-1
    for f in prange(comb_n.shape[0]):
        list_sides=np.zeros((n,n))
        b_new=np.zeros((n,1))
        list_sides[0:n-1]=(hyp_f[comb_n[f]])[:,0:-1]
        list_sides[n-1]=hyperplanes
        b_new[0:n-1]=(hyp_f[comb_n[f]])[:,n:n+1]
        b_new[n-1]=b
        if matrix_rank(list_sides)==n:
        # if stat:
            dum=np.linalg.solve(list_sides,-b_new)
            slope=np.zeros((len(hyp_f),n))
            slope[:,::]=hyp_f[:,0:n]
            if np.argmin(((slope@dum)+hyp_f[:,n:n+1])*sign_m) in comb_n[f]:
                valid_comb[f,:]=dum.T[0]                
    return valid_comb
# @njit(parallel=True)
# def check_combination_p1(comb_n, sides, hyperplane_val, hyp_f, hyperplanes, b, TH, valid_comb, sign_m):
#     n = len(hyp_f[0]) - 1
#     for f in prange(comb_n.shape[0]):
#         base=sides[f[0]]
#         base1=np.zeros((len(base),1))
#         for i in base:
#             for j in f[1:]:
#                 if i in sides[j]:
#                     base1[i]=i
#                 else:
#                     break
#         common=np.array([i for i in base1 if i!=0])
#         if len(common)==2:
#             list_sides=np.zeros((n,n))
#             b_new=np.zeros((n,1))
#             list_sides[0:n-1]=(hyp_f[comb_n[f]])[:,0:-1]
#             list_sides[n-1]=hyperplanes
#             b_new[0:n-1]=(hyp_f[comb_n[f]])[:,n:n+1]
#             b_new[n-1]=b
#             dum=np.linalg.solve(list_sides,-b_new)
#             valid_comb[f,:]=dum.T[0]

#         #     else:
#         #         list_sides = np.zeros((n, n))
#         #         b_new = np.zeros((n, 1))
#         #         list_sides[0:n - 1] = (hyp_f[comb_n[f]])[:, 0:-1]
#         #         list_sides[n - 1] = hyperplanes
#         #         b_new[0:n - 1] = (hyp_f[comb_n[f]])[:, n:n + 1]
#         #         b_new[n - 1] = b
#         #         if matrix_rank(list_sides) == n:
#         #             dum = np.linalg.solve(list_sides, -b_new)
#         #             slope = np.zeros((len(hyp_f), n))
#         #             slope[:, ::] = hyp_f[:, 0:n]
#         #             if np.argmin(((slope @ dum) + hyp_f[:, n:n + 1]) * sign_m) in comb_n[f]:
#         #                 valid_comb[f, :] = dum.T[0]

#         # list_sides[0:n-1] = (hyp_f[comb_n[f]])[:, 0:-1]
#         # list_sides[n-1] = hyperplanes
#         # b_new[0:n-1] = (hyp_f[comb_n[f]])[:, n:n+1]
#         # b_new[n-1] = b

#         # if matrix_rank(list_sides) == n:
#         #     dum = np.dot(inv(list_sides), -b_new)
#         #     if np.argmin(((slope @ dum) + hyp_f[:, n:n+1]) * sign_m) in comb_n[f]:
#         #         valid_comb[f, :] = dum.T[0]

#     return valid_comb
@njit
def check_combination_p1(comb_n, sides, hyperplane_val, hyp_f, hyperplanes, b, TH, valid_comb, sign_m):
    n = len(hyp_f[0]) - 1
    for f in range(comb_n.shape[0]):
        base = np.array(sides[comb_n[f,0]],dtype=np.int64)
        base1 = np.zeros((len(base),1),dtype=np.int64)
        for i in range(len(base)):
            cntr=0
            rem_side=comb_n[f,1:]
            for j in rem_side:
                if base[i] in sides[j]:
                    base1[i] = i
                    cntr=cntr+1
                    if cntr==len(rem_side):
                        break
                else:
                    base1[i] = 0
                    cntr=0
                    break
        common = np.unique(base1)
        # if len(common[1:]) == 2:
        #     if (np.max(hyperplane_val[common[1:]])>=1e-10) and (np.min(hyperplane_val[common[1:]])<=-1e-10):
        #         list_sides = np.zeros((n, n))
        #         b_new = np.zeros((n, 1))
        #         list_sides[0:n - 1] = (hyp_f[comb_n[f]])[:, 0:-1]
        #         list_sides[n - 1] = hyperplanes
        #         b_new[0:n - 1] = (hyp_f[comb_n[f]])[:, n:n + 1]
        #         b_new[n - 1] = b
        #         dum = np.linalg.solve(list_sides, -b_new)
        #         valid_comb[f,:]=dum.T[0]
    return valid_comb
#######################################################################


def finding_inner_intersection(original_polytope_test,boundary_hyperplanes,hyperplanes,b,border_bias,parallel,TH):
    hyperplanes_dum=[]
    index_list=[0]
    bias_dum=[]
    hyperplanes_dum.extend(boundary_hyperplanes[0])
    bias_dum.extend(border_bias[0])
    n=np.shape(original_polytope_test)[2]
    intersections_n=[]
    st=time.time()
    for k in range(len(hyperplanes)):
        hyperplanes_dum.append(hyperplanes[k])
        bias_dum.append(b[k])
        num_comb=math.comb(len(boundary_hyperplanes[0])+k,n-1)
        print("hype:",k+1,"Number of Combination:",num_comb)
        chunk_length=200_000
        n_chunk=num_comb//chunk_length+1
        chunk=[]
        for i in range((n_chunk)):
            if n_chunk>1:
                if i<n_chunk-1:
                    chunk.append([i*chunk_length,(i+1)*chunk_length])
                else:
                    chunk.append([i*chunk_length,num_comb-1])
            else:
                chunk.append([i*chunk_length,num_comb-1])
        # processes=4
        # pool = multiprocessing.Pool(processes=processes)
        comb=combinations(np.arange(len(boundary_hyperplanes[0])+k),n-1)
        array_comb=[]
        cntr=0
        for j,i in enumerate(comb):
            array_comb.append(i)
            if j==chunk[cntr][1]:
                hyp_n=np.array(hyperplanes_dum)
                bias_n=np.array(bias_dum)
                # func = partial(inner_intersection,hyp_n,bias_n,TH)
                comb_n=np.array(array_comb)
                # inner_points =pool.map(func,comb_n)
                if len(array_comb)>200_000 and parallel:
                    valid_comb=2*TH*np.ones((len(comb_n),n))
                    inner_points=inner_intersection_p(hyp_n,bias_n,TH,comb_n,valid_comb,n)
                    inner_points = inner_points[~np.all(inner_points == 2*TH, axis=1)]
                else:
                    inner_points=inner_intersection(hyp_n,bias_n,TH,comb_n)
                     
                array_comb=[]
                cntr=cntr+1
                intersections_n.extend(inner_points)
                if j==chunk[-1][1]:
                    index_list.append(len(intersections_n))
        end=time.time()
        print("Duration:",end-st)
    print("Method2=",end-st)
    return intersections_n,index_list


@njit(parallel=True)
def inner_intersection_p(new_list,b,TH,list1,valid_comb,n): 
    n=np.shape(new_list)[1]
    cntr=0
    # valid_comb=[]
    for i in prange(list1.shape[0]):
        dum0=np.zeros((len(new_list),1))
        id=np.array([[1]*n])
        id[0,0:n-1]=list1[i]
        id[0,-1]=len(new_list)-1
        if matrix_rank(new_list[id[0]])==n:
            dum0[:,0]=b
            dum=np.dot(inv(new_list[id[0]]),-dum0[id[0]])
            if np.linalg.norm(dum, ord=np.inf) <= TH+0.00001:
                valid_comb[i,:]=(dum.T)[0]
                cntr=cntr+1
                # if np.all((dum.T)[0]==np.zeros((n,1))):
                #     stat=True

    return valid_comb

@njit
def inner_intersection(new_list,b,TH,list1):
    n=np.shape(new_list)[1]
    dum0=np.zeros((len(new_list),1))
    valid_comb=[]
    id=np.array([[1]*n])
    for i in list1:
        id[0,0:n-1]=i
        id[0,-1]=len(new_list)-1
        # dum0=np.vstack((new_list[i],new_list[-1]))
        # dum1=np.hstack((b[i,:],b[-1:]))
        if matrix_rank(new_list[id[0]])==n:
            dum0[:,0]=b
            dum=np.dot(inv(new_list[id[0]]),-dum0[id[0]])
            if np.linalg.norm(dum, ord=np.inf) <= TH+0.00001:
                    # If the intersection point is valid, append it to the list
                    valid_comb.append((dum.T)[0])
    # for i in list1:
    #     if matrix_rank(new_list[i])==10:
    #           valid_comb.append(i)
    return valid_comb  


# def finding_valid_intersection(sign_m,hyp_f,intersection_n):


# def check_valid_side_n(valid_side,sides,hyperplane_val,hyp_f,hyperplanes,b,n,TH,parallel,sign_m):
#     lst_new,dum_list=removing_unnecessary_comb(np.array(valid_side),sides,n,hyperplane_val)    
#     len_batch=500_000
#     batch=0
#     # if ln>len_batch:
#     #     batch=int(ln/len_batch)
#     # if ln % len_batch!=0:
#     #     batch=batch+1
#     cntr=0
#     list1=[]
#     intersection_test=[]
#     # if ln >len_batch:
#     #     print("check")
#     # st=time.time()
#     list_final=[]
#     sum=0
#     for j in range(len(lst_new)):
#         batch=0
#         ln=math.comb(len(lst_new[j][1:]),n-2)
#         if ln>len_batch:
#             batch=int(ln/len_batch)
#         if ln % len_batch!=0:
#             batch=batch+1
#         cntr=0
#         list1=np.array(list(combinations(lst_new[j][1:], n-2)))
#         list1=np.insert(list1, 0,lst_new[j][0], axis=1)
#         list_final.extend(list1)
#         if parallel:    
#             if len(list_final)>=len_batch or j==len(lst_new)-1:
#                 if len(list_final)>=10000:
#                     # st=time.time()
#                     # valid_combination=check_combination(np.array(list_final),sides,hyperplane_val,np.array(hyp_f),hyperplanes,b,TH)
#                     # print("dur0:",time.time()-st)
#                     # st=time.time()
#                     valid_combination=check_combination_parallel(TH,np.array(list_final),n,sides,hyperplane_val,np.array(hyp_f),hyperplanes,b,sign_m)                    
#                     # print("Parallel:",time.time()-st)
#                     # st=time.time()
#                     # valid_combination=check_combination(np.array(list_final),sides,hyperplane_val,np.array(hyp_f),hyperplanes,b,TH)
#                     # end=time.time()
#                     # print("GPU:",end-st)
#                     # # st=time.time()
#                 else:
#                     valid_combination=check_combination(np.array(list_final),sides,hyperplane_val,np.array(hyp_f),hyperplanes,b,TH)
#                     del list_final
#                 # valid_combination=check_combination(np.array(list_final),sides,hyperplane_val,np.array(hyp_f),hyperplanes,b,TH)
#                 # if len(valid_combination)!=len(valid_combination1):
#                 #     print("check")
#                 list_final=[]
#                 intersection_test.extend(valid_combination)
#         else: 
#             if len(list_final)>=len_batch or j==len(lst_new)-1:
#                 valid_combination=check_combination(np.array(list_final),sides,hyperplane_val,np.array(hyp_f),hyperplanes,b,TH)
#                 list_final=[]
#                 intersection_test.extend(valid_combination)                
#             del list_final
#             list_final=[]
#             intersection_test.extend(valid_combination)
#     return intersection_test



# # @njit
# def finding_points_regions(intersection_n,sgn_i,n,hyp_f,int_test,TH):
#     new_points=np.hstack((intersection_n,np.ones((len(intersection_n),1))))
#     dum=np.dot(new_points,hyp_f.T)*sgn_i.T
#     int_test=new_function(n,dum,int_test)
#     int_test=int_test[int_test!=-1]
#     return int_test  

# # @njit
# def new_function(n,dum,int_test):
#     for j in prange(len(dum)):
#         dum1=dum[j]
#         if np.min(dum1)>-1e-9:
#             if len(dum1[dum1<=1e-10])==n-1:
#                 int_test[j]=j
#     return int_test
            







# def finding_sign_region(enumerate_poly,n,hyp_f):
#     mid_point=np.reshape(np.mean(enumerate_poly,axis=0),(n,1))
#     if np.all(mid_point==np.zeros((n,1))):
#         mid_point=mid_point+1e-10
#     sign_m=np.sign(np.dot(np.array(hyp_f)[:,0:-1],np.reshape(np.mean(enumerate_poly,axis=0),(n,1)))+np.array(hyp_f)[:,n:n+1])
#     return sign_m



# def finding_all_intersection(sgn_var,boundary_hyperplanes,enumerate_poly,n,intersection_n,TH,border_bias,index_list,i):
#     intersection_wh=[]
#     sum0=0
#     int_test=np.array([-1]*len(np.array(intersection_n[index_list[i]:index_list[i+1]])))
#     for f in range(len(sgn_var)):
#         if sgn_var[f]<-1e-10:
#             sides,hyp_f=finding_side(np.array(boundary_hyperplanes[0]),enumerate_poly[f],np.array(border_bias[0]))
#             sgn_i=finding_sign_region(enumerate_poly[f],n,hyp_f)
#             st=time.time()
#             intersetction_i=finding_points_regions(np.array(intersection_n[index_list[i]:index_list[i+1]]),sgn_i,n,np.array(hyp_f),int_test,TH)
#             end=time.time()
#             sum0=sum0+end-st
#             # intersetction_i = intersetction_i[intersetction_i != -1]
#             intersection_wh.append(intersetction_i)
#     print("finding_sign_region:",sum0)
#     return intersection_wh



# def check_combination_gpu(comb_n,hyp_f,hyperplanes,b,sign_m,n,TH):    
#     valid_comb=cp.asarray(10*TH*np.ones((len(comb_n),n)))
#     n=len(hyp_f[0])-1
#     st=time.time()
#     # device = torch.device('cuda')
#     comb_n=cp.asarray(np.array(comb_n))
#     hyp_f=cp.asarray(np.array(hyp_f))
#     hyperplanes=cp.asarray(hyperplanes)
#     list_sides=cp.zeros((n,n))
#     b_new=cp.zeros((n,1))
#     # valid_comb=torch.FloatTensor(valid_comb).to(device)
#     sign_m=cp.asarray(sign_m)
#     end=time.time()
#     print("Transfer_time:",end-st)
#     for f in range(comb_n.shape[0]):
#         list_sides[0:n-1]=(hyp_f[comb_n[f]])[:,0:-1]
#         list_sides[n-1]=hyperplanes
#         b_new[0:n-1]=(hyp_f[comb_n[f]])[:,n:n+1]
#         b_new[n-1]=b
#         if cp.linalg.matrix_rank(list_sides)==n:
#             dum=cp.matmul(cp.linalg.inv(list_sides),-b_new)
#             if cp.argmin((cp.matmul(hyp_f[:,0:n],dum)+hyp_f[:,n:n+1])*sign_m) in comb_n[f]:
#                 valid_comb[f,:]=dum.T[0]
#     valid_comb=valid_comb[~cp.all(valid_comb == 10*TH, axis=1)]   
#     valid_comb=valid_comb.get()            
#     return valid_comb




# @njit
# def check_combination_test(comb_n,sides,hyperplane_val):
#     valid_comb=[]
#     for f in comb_n:
#         stat=check_sides(f,sides,hyperplane_val)
#         if stat:
#             valid_comb.append(f)
#     return valid_comb




# def check_combination_parallel_n(list1,n,sides,hyperplane_val,hyp_f,hyperplanes,b):
#     list_new=check_combination_test(list1,sides,hyperplane_val)
#     valid_comb1=np.zeros((len(list_new),n))
#     valid_combination=check_combination_pp(np.array(list_new),np.array(hyp_f),hyperplanes,b,valid_comb1)
#     # valid_combination = valid_combination[~np.all(valid_combination == 10*TH, axis=1)]
#     return list(valid_combination)

# @njit(parallel=True)
# def check_combination_pp(comb_n,hyp_f,hyperplanes,b,valid_comb):    
#     n=len(hyp_f[0])-1
#     for f in prange(comb_n.shape[0]):
#         list_sides=np.zeros((n,n))
#         b_new=np.zeros((n,1))
#         list_sides[0:n-1]=(hyp_f[comb_n[f]])[:,0:-1]
#         list_sides[n-1]=hyperplanes
#         b_new[0:n-1]=(hyp_f[comb_n[f]])[:,n:n+1]
#         b_new[n-1]=b
#         valid_comb[f,:]=np.dot(inv(list_sides),-b_new).T[0]                
#     return valid_comb


# @njit
# def finding_common_elements(k,sides):
#     common_elements=[]
#     dum=np.array([-1])
#     for i in k:
#         dum=np.append(dum,sides[i])
#     dum=dum[1:]
#     dum_n=np.unique(dum)
#     for i in dum_n:
#         if len(dum[dum==i])>=len(k):
#             common_elements.append(i)
#         if i==np.max(dum):
#             break
#     return common_elements



def process_polytope(boundary_hyperplanes, enumerate_poly, border_bias, n, hyperplane_val, hyperplanes, b, TH, parallel):
    intersection_test1=[]
    sides,hyp_f=finding_side(np.array(boundary_hyperplanes[0]),enumerate_poly,np.array(border_bias[0]))    
    mid_point=np.reshape(np.mean(enumerate_poly,axis=0),(n,1))
    if np.all(mid_point==np.zeros((n,1))):
        mid_point=mid_point+1e-10
    sign_m=np.sign(np.dot(np.array(hyp_f)[:,0:-1],np.reshape(np.mean(enumerate_poly,axis=0),(n,1)))+np.array(hyp_f)[:,n:n+1])            
    hype_n=np.zeros((len(hyp_f),n+1))
    hype_n[:,:]=np.array(hyp_f)[:,:]
    
    # if len(intersection_n)<5000:
    #     st=time.time()
    #     intersection_test_n=func_n(hype_n,np.array(intersection_n),sign_m,n)
    #     print("Duration_jit:",time.time()-st)
    # else:
    # st1=time.time()
    # intersection_test_nn=func_torch(hype_n,np.array(intersection_n),sign_m,n)
    # print("Duration_torch:",time.time()-st1)
    # else:
    #     intersection_test1=10*TH*np.ones((len(intersection_n),n))
    #     # sign_m=sign_m.repeat(len(intersection_n),axis=1)    
    #     st10=time.time()
    #     intersection_test1=func_np(hype_n,np.array(intersection_n),sign_m,n,intersection_test1)
    #     print("Duration_parallel:",time.time()-st10)
    #     intersection_test_n = intersection_test1[~np.all(intersection_test1 == 10*TH, axis=1)]
    #     del intersection_test1

    # print("Duration:",time.time()-st)
    st1=time.time()
    intersection_test=finding_valid_side(sides,np.array(hyperplane_val),n,hyp_f,hyperplanes,b,TH,parallel,sign_m)
    # print("Duration1:",time.time()-st1)
    del sides,hyp_f
    if len(intersection_test)<n-1:
        raise Warning("check: Number of valid sides is less than $n-1$")
    original_polytope_test=Polytope_formation_hd(enumerate_poly,np.array(hyperplane_val),TH,np.array(intersection_test))
    del intersection_test

    return original_polytope_test


@njit
def tailoring_lists(list_initial,dum_list):  
    list_final=[ ]
    for i in list_initial:
        stat=True
        for j in dum_list:
            if (j[0] not in i) or (j[1] not in i):
                stat=True*stat
            else:
                stat=False
                break
        if stat==True:
            list_final.append(i)
    return list_final

@njit(parallel=True,fastmath=True)
def tailoring_lists_parallel(list_initial, dum_list):  
    results = np.zeros(np.shape(list_initial), dtype=np.int32)
    # result_count = 0
    
    for idx in prange(len(list_initial)):
        i = list_initial[idx]
        stat = True
        for j in dum_list:
            if (j[0] not in i) or (j[1] not in i):
                continue
            else:
                stat = False
                break
        if stat:
            results[idx,:] = list_initial[idx]
            # result_count += 1
    
    # list_final = [list_initial[i] for i in results[:result_count]]
    return results


@njit
def func_n(hype_n,intersection_n,sign_m,n):
    intersection_test1=[]
    slope=np.zeros((len(hype_n),n))
    slope[:,:]=hype_n[:,0:n]
    test=sign_m*(slope@intersection_n.T+hype_n[:,n:n+1])
    for i in range(np.shape(test)[1]):
        if np.min(test[:,i])>-1e-8:
            intersection_test1.append(intersection_n[i])
    return intersection_test1

def finding_deep_hype(hyperplanes,b,S_prime,border_hyperplane,border_bias,i,n):
    S_init=np.eye(np.shape(hyperplanes[0])[0])
    hype=S_init@hyperplanes[0]
    bias=(S_init@b[0])

    # border_hyperplane=np.vstack((border_hyperplane,hyperplanes[0]))
    # border_bias=border_bias+b[0].tolist()
    m=2*n
    for j in range(0,i):
        border_hyperplane=np.vstack((border_hyperplane,hype))
        border_bias=np.hstack((np.array(border_bias),bias))
        mid_point=np.mean(S_prime,axis=0)
        S=np.sign(np.maximum(border_hyperplane@mid_point+border_bias,0))
        hype=hyperplanes[j+1]@np.diag(S[m:])@hype    
        bias=hyperplanes[j+1]@np.diag(S[m:])@bias+b[j+1]
        m=m+(np.shape(hyperplanes[j])[0])
    return hype,bias,border_hyperplane,border_bias.tolist()




@njit(parallel=True)
def func_np(hype_n,intersection_n,sign_m,n,intersection_test1):
    slope=np.zeros((len(hype_n),n))
    slope[:,:]=hype_n[:,0:n]
    for i in prange(np.shape(intersection_n)[0]):
        test=sign_m*(slope@intersection_n[i:i+1,:].T+hype_n[:,n:n+1])
        if np.min(test)>-1e-8:
            intersection_test1[i]=intersection_n[i]
    return intersection_test1


def func_torch(hype_n,intersection_n,sign_m,n):
    device = torch.device('cuda')
    # intersection_test1=torch.FloatTensor(intersection_test1).to(device)
    hype_n=torch.tensor(hype_n,dtype=torch.float64,device=device)
    intersection_n=torch.tensor(intersection_n,dtype=torch.float64,device=device)
    sign_m=torch.tensor(sign_m,dtype=torch.float64,device=device)
    # slope=torch.zeros((len(hype_n),n)).to(device)
    # slope[:,:]=hype_n[:,0:n]
    test=torch.min(sign_m*(torch.matmul(hype_n[:,0:n],intersection_n.T)+hype_n[:,n:n+1]),dim=0).values
    indices=test>-1e-8
    intersection_test1=intersection_n[indices]
    # for i in range(np.shape(test)[1]):
    #     if torch.min(test[:,i])>-1e-8:
    #         intersection_test1[i]=intersection_n[i]
    return intersection_test1.cpu().numpy()




def GPU_Intersection(intersection_n,boundary_hyperplanes,border_bias,signs):
    signs=np.array(signs)
    device = torch.device('cuda')
    intersection_nn=torch.tensor(intersection_n,dtype=torch.float64,device=device)
    signs_gpu=torch.tensor(signs,dtype=torch.float64,device=device)
    signs_gpu=signs_gpu.unsqueeze(1)
    border_bias_gpu=torch.tensor(np.array(border_bias[0]).reshape((len(border_bias[0]),1)),dtype=torch.float64,device=device)
    boundary_hyperplanes_gpu=torch.tensor(boundary_hyperplanes[0],dtype=torch.float64,device=device)
    
    
    all_values=torch.matmul(boundary_hyperplanes_gpu,intersection_nn.T)+border_bias_gpu 
    sgn_var_gpu=signs_gpu*all_values
    min_value=torch.min(sgn_var_gpu,dim=0).values
    # min_value = torch.where(min_value > -1e-8, torch.tensor(1.0, dtype=torch.float64, device=device), torch.tensor(0.0, dtype=torch.float64, device=device))
    indices=min_value>-1e-8
    res=intersection_nn[indices]
    # min_value[min_value>-1e-6]=1.0
    # min_value[min_value<-1e-6]=0.0
    # m, n = min_value.shape

    # Initialize the output matrix
    # block_diagonal_tensor = torch.diag_embed(min_value)
    # block_diagonal_tensor = torch.zeros((m, n, n), dtype=min_value.dtype,device=device)

    # Generate indices for placing diagonal elements
#     rows = torch.arange(m).repeat_interleave(n)
#     cols = torch.arange(n).repeat(m)
#     diag_indices = rows * n + cols

#     # Flatten the input matrix
#     flat_matrix = min_value.flatten()

# # Fill the block diagonal tensor
#     block_diagonal_tensor[rows, cols, cols] = flat_matrix
    # res=torch.matmul(block_diagonal_tensor,intersection_nn)
    # print(res)
    return res.cpu().numpy().tolist()



# def compute_polytope_vertices(
#     A: np.ndarray, b: np.ndarray
# ) -> list[np.ndarray]:
#     r"""Compute the vertices of a polytope.

#     The polytope is given in halfspace representation by :math:`A x \leq b`.

#     Parameters
#     ----------
#     A :
#         Matrix of halfspace representation.
#     b :
#         Vector of halfspace representation.

#     Returns
#     -------
#     :
#         List of polytope vertices.

#     Notes
#     -----
#     This method won't work well if your halfspace representation includes
#     equality constraints :math:`A x = b` written as :math:`(A x \leq b \wedge
#     -A x \leq -b)`. If this is your use case, consider using directly the
#     linear set ``lin_set`` of `equality-constraint generatorsin pycddlib
#     <https://pycddlib.readthedocs.io/en/latest/matrix.html>`_.
#     """
#     b = b.reshape((b.shape[0], 1))
#     mat = cdd.Matrix(np.hstack([b, -A]), number_type="float")
#     mat.rep_type = cdd.RepType.INEQUALITY
#     P = cdd.Polyhedron(mat)
#     g = P.get_generators()
#     V = np.array(g)
#     vertices = []
#     for i in range(V.shape[0]):
#         if V[i, 0] != 1:  # 1 = vertex, 0 = ray
#             raise ValueError("Polyhedron is not a polytope")
#         elif i not in g.lin_set:
#             vertices.append(V[i, 1:])
#     return vertices
