import cProfile
import pstats
from relu_region_enumerator import enumeration_function
# from Enum_module_old import enumeration_function
mode="Rapid_mode" 
parallel=True
import torch
# mode="Low_Ram"
# from memory_profiler import profile
if __name__=='__main__':
    with cProfile.Profile() as pr:
        # NN_file="NN_files/test_simple_6d_40.xlsx"
        # NN_file="NN_files/Path_following_20.xlsx"
        # NN_file="NN_files/model_6d_20_20_deep.pt"
        # NN_file="NN_files/model_quadrotor_modified.pt"
        # NN_file="NN_files/model_6d_dee.pt"
        NN_file="NN_files/model_decay_2_12_n.pt"
        eps1=0.001
        eps2=0.01
        name="decay"
        TH=TH = [2.0]*6
        barrier_model = torch.jit.load("NN_files/model_decay_2_12_n.pt", map_location="cpu")
        barrier_model.eval()
        enumeration_function(NN_file,name,TH,mode,parallel,verification='barrier',barrier_model=barrier_model)
    stats=pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)    
    stats.dump_stats(filename='34dprofiling.prof')


