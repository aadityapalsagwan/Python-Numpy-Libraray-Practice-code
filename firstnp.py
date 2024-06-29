import numpy as np

# x = np.array([1,2,3,4,5]) # Create a numpy array using list 
# print(type(x))  
# print(x)
# y = [1,2,3,4,5]
# print(y)
# print(type(y))


# %timeit [j**4 for j in range(1,9)]  # 2.88 µs ± 166 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

# %timeit np.arange(1,9)**4   # 2.26 µs ± 142 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

# ======================== Array ================

# x = [1,2,3,4,5]
# y = np.array(x)
# print(y)

# ======================== Create Array using a for loop input ================

# l = []
# for i in range(1,6):
#     int_1 = int(input("Enter num: "))
#     l.append(int_1)

# print("This is your List: ",l)
# print("This your np Array: ",np.array(l))    
# print(l.ndim)


# ============================ create 2d Array ==================

# ar2 = np.array([[1,2,3,4],[4,5,6,7]])
# print(ar2)
# print(ar2.ndim)

# ============================ create 3d Array ==================

# ar3 = np.array([[[1,2,3],[4,5,6],[7,8,9]]])
# print(ar3)
# print(ar3.ndim)

# ============================ create nth d Array ==================

# arn = np.array([1,2,3,4], ndmin = 20)
# print(arn)
# print(arn.ndim)

# ========== zero (0) array ==================

# ar_Zero = np.zeros(4)
# print()
# ar_Zero1 = np.zeros((3,4)) # 2d array
# print(ar_Zero)
# print(ar_Zero1)

# ========== one (1) array ==================

# ar_one = np.ones(4)
# print()
# ar_one1 = np.ones((3,4)) # 2d array
# print(ar_one)
# print(ar_one1)

# ========== empty () array ==================

# ar_em = np.empty(4)
# print("Done: ",ar_em)

# ========== Range () array ==================

# ar_rn = np.arange(4)
# print(ar_rn)

# ========== Diagonal array ==================

# ar_dia = np.eye(4)
# print(ar_dia)

# ========== linspace array ==================

# ar_line = np.linspace(1,10,num=5)
# print(ar_line)

# ================ make a array using a Random number ( rand(), randn(), ranf(), randint()) ==========================

# var = np.random.rand(4)
# var = np.random.randn(4)  # give a - or + value in the array.
# var = np.random.ranf(4)  # give a float value in the array.
# var = np.random.randint(4,20,5)  
# print(var)

# ================= Data Type of Numpy Array =================

# var = np.array([1,5,9,7,6,7,4])
# print("Data type: ",var.dtype)

# var = np.array([10.2,25.8,9.5])
# print("Data type: ",var.dtype)

# var = np.array(["A","d","i"])
# print("Data type: ",var.dtype)

# var = np.array(["A","d","i",1,5,9.3,80.6])
# print("Data type: ",var.dtype)

# x = np.array([1,5,9,6],dtype = np.int8) # convert the data type int32 to int8 function
# x = np.array([1,5,9,6],dtype = "f")
# x = np.array([1,5,9,6],dtype = "U")  # string 
# print(x)
# print("Data type: ",x.dtype)

# x = np.array([2,1,9,6,3])
# new = x.astype(float) # change a data type of array using function..
# print(x)
# print(new)

# ================ Shape and Reshape Array in Numpy ====================

# var = np.array([[1,2],[3,4]])
# print(var)
# print()
# print(var.shape)  # its means 2*2 ki matrix

# ============ reshape ================

# var = np.array([1,2,3,4,5,6])
# x=var.reshape(3,2)  #its means 1-dimentional change to any dimentional...
# print(x)
# print()
# one = x.reshape(-1)
# print(one)     # its means change the original form.... 

# ================================ Arithmetic Operation in NUmpy Array ========================
#   1-D
# var = np.array([1,2,3,4])
# varadd = var + 3
# print(varadd)

#    2-D Array

# var = np.array([[1,2,3,4],[5,6,7,8]])
# var1 = np.array([[1,2,3,4],[5,6,7,8]])
# varadd = var + var1
# print(varadd)

# ====================== Airthmetic Function =========================

# var = np.array([1,2,3,4,5,3,2])
# print("Minimum is: ",np.min(var),np.argmin(var))
# print("Maximum is: ",np.max(var),np.argmax(var))

# var = np.array([1,2,3])
# print("Sin value is: ",np.sin(var))
# print("cos value is: ",np.cos(var))
# print("square value is: ",np.sqrt(var))
# print("cumsum value is: ",np.cumsum(var))

# ============ BroadCasting Numpy Array ===================

# var = np.array([1,2,3,4])
# var1 = np.array([1,2,3])
# print(var + var1)

# var = np.array([[1],[2]])
# print(var.shape)
# var1= np.array([[1,2],[3,4]])
# print(var1.shape)
# print()
# print(var+var1)

# ================= Indexcing and Slicing in Numpy Array =====================

# Indexing 1D Aaray

# var =np.array([9,8,7,6])
# print(var[3])

# 2D Array Indexing 

# var = np.array([[9,8,7],[6,5,4]])
# print(var)
# print(var.ndim)
# print()
# print(var[1,0])

# 3D Array Indexing 

# var = np.array([[[1,2],[3,4]]])
# print(var)
# print()
# print(var.ndim)
# print()
# print(var[0,0,1])

# Slicing Numpy Array 1D array

# var = np.array([1,2,3,4,5,6,7])
# print(var)
# print()
# print("2 to 5 :- ",var[1:5])
# print("Stop/ jump:- ",var[1:6:2])

# 2D Array Slicing.... 

# var = np.array([[1,2,3,7,8,9],[4,5,6,1,2,3],[7,8,9,4,5,3]])
# print(var) 
# print()
# print("5 to 3 :- ",var[1,1:])


# =========================================== Iterating NumPy Array ===============================

# 1D Array iteration 

# var = np.array([1,2,3,4,8,9])
# print(var)
# print()
# for i in var:
#     print(i)
    
# 2D Array iteration 

# var = np.array([[4,5,6,9],[1,8,3,7]])
# print(var)
# print(var.ndim)
# print()

# for i in var:
#     print(i)
# print()
# for i in var:
#     for l in i:
#         print(l)
#     print()

# for i in np.nditer(var):        # no multi loop are using for iteration... 
#    print(i)
    
# for i,d in np.ndenumerate(var):   # index number are involve in iteration.... 
#    print(i,d)    

# ======================================= Copy vs View in NumPy Array =================================

# var = np.array([4,5,3,1])
# co = var.copy()
# var[2]=40
# print("Var : ",var)
# print("Copy : ",co)

# var = np.array([9,7,6,1,5,3])
# vi = var.copy()
# var[3]=50
# print("Var : ",var)
# print("View : ",vi)

# =============================== Join and Split Function in NumPy Array ===============================

# x = np.array([4,5,3,1,2])
# y = np.array([7,8,9,3,4,6])
# print(x)
# print(y)
# print()
# var = np.concatenate((x,y))
# print(var)

# 2D Array 

# x = np.array([[1,2],[3,4]])
# y = np.array([[5,6],[7,8]])
# print(x)
# print(y)
# print()
# var = np.concatenate((x,y),axis=1)
# print(var)

# Using Stack function... 

# x = np.array([4,5,3,1,2])
# y = np.array([7,8,9,3,4])
# print(x)
# print(y)
# print()
# var = np.stack((x,y),axis = 1)
# print(var)

# Split Array function... 

# y = np.array([7,8,9,3,4,9,5,6,4,3])
# x = np.array([[7,8],[9,3],[4,9],[5,6],[4,3]])
# print(y)
# var = np.array_split(y,3)
# var1 = np.array_split(x,3,axis=1)
# print()
# print(var)
# print(var1)
# print(type(var))

# ======================= Search , Sort, Search Sorted, Filter in NumPy Array ===================

# Search 

# var = np.array([1,2,3,4,2,5,2,5,6,7])
# print(var)
# # x = np.where(var == 2)
# x = np.where((var%2) == 0)
# print(x)

# Search Sorted Array 

# var = np.array([1,2,3,4,5,6,8,9])
# print(var)
# x = np.searchsorted(var,7)
# print("The index of this value: ",x)

# Sort Function 

# var = np.array([1,2,13,54,2,65,22,5,96,7])
# vr = np.array(["a","i","d"])
# print(var)
# print()
# print(np.sort(var))
# print(np.sort(vr))

# Filter Array 

# x = np.array(["a","t","p","r","s"])
# f = [True,False,True,False,True]
# new = x[f]
# print(new)

# ======================== Suffle Unique Resize Flatten Ravel in NumPy Array =======================

# var = np.array([1,2,3,4,5])
# np.random.shuffle(var)
# print(var)

# var = np.array([1,2,6,2,5,3,9,3,8,2,3,4])
# x = np.unique(var,return_index=True)
# x = np.unique(var)
# print(x)

# var = np.array([1,2,3,4,5,6])
# x = np.resize(var,(2,3))
# print(x)

# var = np.array([1,2,3,4,5,6])
# x = np.resize(var,(2,3))
# print(x)
# print()
# print("Flattern :- ",x.flatten(order="F"))
# print("Ravel :- ",np.ravel(var,order="F"))

# ================================ Insertion And Delete Function in NumPy Array =======================

# Insert 

# var = np.array([1,2,3,4])
# print(var)
# # v = np.insert(var,2,20)
# v = np.insert(var,(1,2),[20,85])  # alag alga position pr ek he value.. 
# print()
# print(v)

# Delete 

# var = np.array([1,2,3,4])
# print(var)
# d = np.delete(var,1)
# print(d)

# ================================ Matrix in NumPy Array ===========================

# var = np.matrix([[1,2],[4,5]])
# var1 = np.matrix([[1,2],[4,5]])
# print(var)
# print()
# print(var+var1)
# print()
# print(var-var1)
# print()
# print(var*var1)
# print()
# print("Dot : ",var.dot(var1))

# ============================== matrix Function in NumPy Array =====================

# var = np.matrix([[1,2,3],[4,5,6]])
# print(var)
# print()
# print(np.transpose(var))

var = np.matrix([[1,2],[3,4]])
print(var)
# print(np.swapaxes(var,0,1))
print()
# print(np.linalg.inv(var))
print()
# print(np.linalg.matrix_power(var,2))
# print(np.linalg.matrix_power(var,0))
# print(np.linalg.matrix_power(var,-2))
print(np.linalg.det(var))  # determinate matrix