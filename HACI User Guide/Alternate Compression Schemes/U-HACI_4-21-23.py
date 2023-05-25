#!/usr/bin/env python
# coding: utf-8

# In[2]:




import numpy as np
import matplotlib.pyplot as plt


# In[12]:

try:
	CI_1=np.fromfile(r'CI_1_1.dat')
except:
	CI_1=np.fromfile(r'CI_3_1.dat')
size=int(np.sqrt(np.shape(CI_1)))
print(size)


# In[13]:


CI_1_sq = np.reshape(CI_1, (-1,size))
absCI_1=np.log10(abs(CI_1_sq))
print(absCI_1)
print(np.shape(absCI_1))


"""Compressed matrix.

Hierarchical matrix implementation.

"""

import numpy as np
import scipy.sparse.linalg as lg
from random import randrange

import string
import random

class CHMatrix():
    """CMAtrix class that represents a matrix or a block"""

    def __init__(self, mat,type,level,threshold):
        """
        CMAtrix constructor.

        Args:
            mat (ndarray) : Matrix to compess.
        """

        # Store the number of rows and columns
        self.nr = mat.shape[0]
        self.nc = mat.shape[1]

        self.type = type
        self.level= level
        self.threshold=threshold

        # Ranks
#        k_svd   = 6*self.level**3+1
#        print("k = ",k_svd)
        k_dense = 6

        # Make small blocks dense
        if min(self.nr, self.nc) <= k_dense:
            self.type = 0

        #For the Hierarchical part, we use typ==0 to indicate the dense matrix condition
        #when length no more than 8. typ==3 indicates SVD when the matrix is far from diagonal.
        #typ equals 2,1 and -1 respectively indicates the diagonal, upper and lower blocks.
            
            
        if self.type == 0:
            self.mat = mat

        
        elif self.type == 3:
            tst=0
            rk=1
            while tst < 4:
                u, s, vt = lg.svds(mat, k=rk) # SVD rank is 6 by default
                if np.min(s)**2 < 2*self.nr*threshold:
                    tst = 10
                    nuv=0
                    for i in range(rk):
                        if s[i]**2 > 2*self.nr*threshold:
                            nuv=nuv+1
                    
                    self.nuv = nuv
                    if 2 * nuv < self.nr:
                        self.u  = u[:,rk-nuv:rk]
                        self.s  = s[rk-nuv:rk]
                        self.vt = vt[rk-nuv:rk,:]
                    else :
                        self.type = 0
                        self.mat=mat

                else :
                    tst = 0
                    if rk*2 < self.nr:
                        rk = rk*2
                    else:
                        self.mat=mat
                        self.type=0
                        tst=10
            self.rk=rk

        #changed to initially svd all but the first block 
        #H-mat first block
        elif self.type == 2:
            i = self.nr // 2
            j = self.nc // 2
            self.b11 = CHMatrix(mat[:i,:j],2,self.level+1,self.threshold)
            self.b12 = CHMatrix(mat[:i,j:],3,self.level+1,self.threshold)
            self.b21 = CHMatrix(mat[i:,:j],3,self.level+1,self.threshold)
            self.b22 = CHMatrix(mat[i:,j:],3,self.level+1,self.threshold)



    def dot(self,x):
        """
        Matrix-vector multiplication.

        Args:
            x (ndarray) : Initial vector.

        Returns:
            y (ndarray) : Resulting vector.
        """

        # Check if matrix and vector sizes mismatch
        if self.nc != len(x):
            print('Matrix-vector size mismatch')
            sys.exit(1)

        # Dense multiplication
        if self.type == 0:
            y = self.mat.dot(x)

        # Or multiplication using SVD decomposition
        elif self.type == 3:
            sigma = np.diagflat(self.s)  # Form a diagonal matrix from vector S
            y = self.u.dot(sigma.dot(self.vt.dot(x)))

        # Or delegate to sub-blocks and combine pieces
        else:
            j  = self.nc // 2
            y1 = self.b11.dot(x[:j]) + self.b12.dot(x[j:])
            y2 = self.b21.dot(x[:j]) + self.b22.dot(x[j:])
            y  = np.concatenate([y1,y2])

        return y
###########################################################################################################
    #make rebuilder iterable like bstr
    def rebuilder(self):
        """
        Rebuilds an uncompressed version of the given matrix
        """
        
        
        # type==0 for dense matrix condition, type==3 for svd, type==2 diagnal, type==1,-1 for upper and lower diagnal
        if self.type == 0:
            Block = self.mat
            #print("Block type = 0",Block)
            #print(type(Block),Block.shape)


            
        elif self.type == 3:
            sigma = np.diagflat(self.s)  # Form a diagonal matrix from vector S
            
            Block = np.dot(self.u,np.dot(sigma,self.vt))
            #print("Block type = 3",Block)
            #print(type(Block),Block.shape)

        else:

            b1 = np.concatenate((self.b11.rebuilder(), self.b12.rebuilder()), axis=1)
            b2 = np.concatenate((self.b21.rebuilder(), self.b22.rebuilder()), axis=1)
            Block  = np.concatenate((b1, b2), axis=0)
            #print("Block type = else",Block)
            #print(type(Block),Block.shape)

            
            
            
            
            
        return Block
    
    def __str__(self):
        """
        Returns the reconstructed version of the matrix.
        """

        Block  = ''
        Block += ''
        for i in self.rebuilder():
            Block += '    '.join(str(j) for j in i)
            Block += '\n'
        
        return Block
############################################################################################################     





    def memory(self):
        """
        Computes the size of memory used.

        Returns:
            k (int) : The number of doubles stored.
        """

        # Return the number of elements if dense
        if self.type == 0:
            k = self.nr * self.nc

        # Or the number of doubles in SVD decomposition
        elif self.type == 3:
            k  = self.u.shape[0] * self.u.shape[1]
            k += self.s.shape[0]
            k += self.vt.shape[0] * self.vt.shape[1]

        # Or sum over sub-blocks
        else:
            k1 = self.b11.memory() + self.b12.memory()
            k2 = self.b21.memory() + self.b22.memory()
            k  = k1 + k2

        return k


    def error(self,mat):
        """
        Computes matrix error.

        Generates a number of random vectors, multiplies by the matrix
        and computes the residual norms. The error is relative and is defined as
        a ratio of the residual norm and the norm of the exact solution. 
        The final error is averaged over all random vectors.

        Args:
            mat (ndarray) : The initial full matrix that is approximated.

        Returns:
            e (double): Error.
        """
    
        count = 1000
        e = 0

        for i in range(count):
            x  = np.random.rand(n)
            yd =  mat.dot(x)
            yc = self.dot(x)
            dt = yd - yc
            e += np.linalg.norm(dt) / np.linalg.norm(yd)
    
        e /= count
    
        return e


if __name__ == "__main__":

    # Matrix size
    
    # Matrix size
    n = size

    # Generate a random symmetric matrix
    #mat = np.random.rand(n,n)
    #mat = (mat + mat.T) / 2
    #for i in range(n):
    #    for j in range(n):
    #        mat[i][j]=mat[i][j]*10/(abs(i-j)+1)
    mat=CI_1_sq

    #print('Given matrix:')
    #print(mat)
    print()

    # Generate a random vector
    #print('Given vector:')
    x = np.random.rand(n)
    #print(x)
    print()

    # Dense matrix vector multiplication
    #print('Matrix vector product [Dense]:')
    y = mat.dot(x)
    
    #print(y)
    #print()

    # Generate a compressed matrix that approximates the given matrix

    
    newmat=np.empty
    #print(newmat)
    #print('CMatrix:')
    thres=7e-7
    print("Thresh =", thres)
    cmat=CHMatrix(CI_1_sq,2,0,thres)
    print("memory of CHMatrix is")
    print(cmat.memory())
    np.append(newmat,cmat)
    #print(cmat)
    print()
    
    #print(newmat,"New")

    # Compressed matrix vector multiplication
    #print('Matrix vector product [CMatrix]:')
    y = cmat.dot(x)
    #print(y)
    #print()

    # Format strings for printing
    fs = '{0:10s} {1:10d} {2:10.5f}'
    fd = '{0:10d} {1:10d} {2:10.5f}'

    # Print the information about both matrices
    #print("k ,",k_svd)
    print('      Name     Memory   RelError')
    kd = n * n
    kc = cmat.memory()
    ed = 0
    ec = cmat.error(mat)
    print(fs.format('Dense', kd, ed))
    print(fs.format('CMatrix', cmat.memory(), ec))
    print()
    print()




# In[17]:


#print(cmat)
k=cmat

k=str(k)
ciL=list(map(float,k.split()))
CiA=np.array(ciL)
CiA=CiA/np.linalg.norm(CiA)
ci=CiA.tolist()
#print(ci)
#print(np.dot(CiA,CiA),"Normilization Check")


# In[37]:



import struct
import numpy
f=open("recn_init.bin","wb")
bin=struct.pack('d'*len(ci),*ci)
#print(bin)

f.write(bin)
f.close()

print("done")



# In[ ]:




