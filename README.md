# Enumerttion Algorithm
In this Code, an enumeration algorithm using vertices and hyperplanes is developed.

## Input
The input to this algorithm is a single hidden layer ReLU Neural Netweork. We consider the single hidden layer ReLU as $\dot{x}=W\sigma_.(Hx+b)+c$ where $x\in R^n$. We need to store all the weights and biases of the networks in a *'.xlsx'* file. In this *'.xlsx'* file, the data stored as follows:

* $H \in R^{m*n}$  where $m$ is the number neurons must be stored in sheet 1.
* $b\in R^{m*1}$ must be stored in sheet 2.
* $W \in R^{n*m}$ must be stored in sheet 3.
* $c \in R^{n}$ must be stored in sheet 4.

**Note:** In the `Neural` folder, a code is availbale, `nn_simple.py`, to identify a dynamical system using a single-hidden Layer ReLU and saving the data as described above. 
## How to run the code?
In order to run the code, 

`>>python3 .\script.py`

 should be run in the terminal. In this script file, `NN_file="path\NN_file.xlsx"` will clarify the desired NN that must be considered for enumartion.
 `name="output_file"` will determine the name of the output file. All the related information, will be stored in `cwd:\Results\output_file.m` by its vertices.










 Some of the results were showed in the following table.
| Example       | Hyperplanes   | cells      | Time(s)     |
| ------------- | ------------- |------------|-------------|
| 3D            | 10            |134         |0.04         |
| 3D            | 100           |45563       |49           |
| 4D            | 10            |905         |0.5          |
| 4D            | 74            |563553      |300          |
| 4D            | 104           |1972117     |1302         |
| 6D            | 56            |3804757     |14088        |
