from Enumeration_module import enumeration_function
# from Enumeration_module import enumeration_function
# from Enum_module_old import enumeration_function
mode="Rapid_mode" 
parallel=True
# mode="Low_Ram"
# from memory_profiler import profile
if __name__=='__main__':
    # with cProfile.Profile() as pr:
        # NN_file="NN_files/test_simple_6d_40.xlsx"
        # NN_file="NN_files/Path_following_20.xlsx"
    # NN_file="NN_files/model_IP_Pedram_test_deep.pt"
    NN_file="NN_files/model_IP_Pedram_test_deep.pt"
    eps1=0.001
    eps2=0.01
    name="model_8d_deep"
    TH=[3.14,3.14]
    
    enumeration_function(NN_file,name,TH,mode,parallel)
