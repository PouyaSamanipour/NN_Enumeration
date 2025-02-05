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
- `pycddlib`
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

$\mathcal D=\{x\in \mathbb R^n:|x|_\infty\leq TH\}$. 

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

