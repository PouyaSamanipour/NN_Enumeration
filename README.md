<!-- # Enumerttion Algorithm
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
| 6D            | 56            |3804757     |14088        | -->
<!-- <!-- # Enumerttion Algorithm
In this Code, an enumeration algorithm using vertices and hyperplanes is developed.

## Input
The input to this algorithm is a single hidden layer ReLU Neural Netweork. We consider the single hidden layer ReLU as $\dot{x}=W\sigma_.(Hx+b)+c$ where $x\in R^n$. We need to store all the weights and biases of the networks in a *'.pt'* file. 


**Note:** In the `Neural` folder, a code is availbale, `nn_simple.py`, to identify a dynamical system using a single-hidden Layer ReLU and saving the data as described above. 
## How to run the code?
In order to run the code, 

`>>python3 .\script.py`

 should be run in the terminal. In this script file, `NN_file="path\NN_file.xlsx"` will clarify the desired NN that must be considered for enumartion.
 `name="output_file"` will determine the name of the output file. All the related information, will be stored in `cwd:\Results\output_file.m` by its vertices.










 Some of the results were showed in the following table.
| Example       | Hyperplanes   | cells      | Time(s)     |
| ------------- | ------------- |------------|-------------|
| 6D            | 20-20         |26733       |46           | -->


<!-- # **Vertex-based Enumeration Algorithm for ReLU NNs**

This package provides an **enumeration algorithm** that computes the partitioning of the input space induced by a **Deep ReLU Neural Network (ReLU NN)**. The algorithm extracts **hyperplanes** from the network and efficiently enumerates the corresponding cells in the partition.

---

## **Installation**
To install this package, first, clone the repository and navigate to the package directory:

```sh
git clone https://github.com/PouyaSamanipour/vertex_enum_relu.git
cd vertex_enum_relu --> -->

# **Vertex-based Enumeration Algorithm for ReLU NNs**

This package provides an **enumeration algorithm** that computes the partitioning of the input space induced by a **Deep ReLU Neural Network (ReLU NN)**. The algorithm extracts **neurons or hyperplanes** from the network and efficiently enumerates the corresponding cells in the partition.

---

## **Installation**
To install this package, first, clone the repository and navigate to the package directory:

```sh
git clone https://github.com/PouyaSamanipour/NN_Enumeration.git
# cd vertex_enum_relu
```

Then, install the package along with its dependencies using:

```sh
pip install Enumeration_module
```

<!-- This will install the package in **editable mode**, allowing you to modify the code without reinstalling. -->

<!-- Alternatively, if the package is published on PyPI, install it directly with:

```sh
pip install Enumeration_module
```

--- -->

## **Dependencies**
The package requires the following dependencies, which will be installed automatically with `pip install -r requirements.txt`:

- `numpy`
- `numba`
- `torch`
- `scipy`
- `matplotlib`
- `cddlib`
- `pandas` (for CSV handling)

If not installed automatically, you can manually install them using:

```sh
pip install numpy numba torch scipy matplotlib cddlib pandas
```

---

## **How to Use**
### **1. Running the Enumeration Algorithm**
To run the enumeration algorithm using a saved **single hidden-layer ReLU neural network**, execute:

```sh
python3 script.py
```

### **2. Specifying the Neural Network File**
Inside `script.py`, specify the **path to the trained ReLU NN** stored in a `.pt` file. The network is assumed to follow the form:

$\dot{x} = W\sigma(Hx + b) + c $

In this file, $TH$ is the threshold of the domain that we cosidered enymeration. Our domain in this version is defined as:

$\mathcal D=\{x\in \mathbb R^n:||x||_\infty\leq TH\}$. 

**"Note: It is recommended to set `parallel=True` in complex examples."**  

Modify the `NN_file` variable in `script.py` to point to the desired **Neural Network file**:

```python
NN_file = "path/to/NN_file.pt"
```

<!-- ### **3. Setting the Output File Name**
Define the name of the **output file** where results will be stored:

```python
name = "output_file"
```

All results, including the **enumerated vertices**, will be stored in:

```
cwd:/Results/output_file.m
```

--- -->

<!-- ## **Using Neural Network Identification**
The package includes a **Neural Network Identification** script in the `Neural/` folder:

- **`nn_simple.py`**: This script helps in **training and saving a single-hidden-layer ReLU network** for a given dynamical system.
- The trained network is stored in a **`.pt`** file, which can be used as input for the enumeration algorithm.

--- -->

## **Example Results**
Below is an example of the enumeration results, showing the number of **hyperplanes, computed cells, and execution time** for different neural network models.

| Example       | Hyperplanes   | Cells      | Time (s)     |
|--------------|--------------|------------|-------------|
| 6D System    | 20-20        | 26,733     | 46          |

<!-- ---

## **Future Work**
- Extending the enumeration algorithm to **deep ReLU networks**.
- Improving efficiency using **parallel computation**.
- Adding support for **more complex dynamical systems**.

--- -->

## **License**
This package is released under the **MIT License**.

For any questions or contributions, feel free to contact **Pouya Samanipour** at **psa254@uky.edu**.

