#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
import struct
import numpy

import numpy as np
import scipy.sparse.linalg as lg
from random import randrange

import string
import random




k_svd   = 2





try:
	CI_1=np.fromfile(r'CI_1_1.dat')
except:
	CI_1=np.fromfile(r'CI_3_1.dat')
size=int(np.sqrt(np.shape(CI_1)))
CI_1_sq = np.reshape(CI_1, (-1,size))
#print(CI_1_sq)
print(np.linalg.norm(CI_1_sq))
print()

u, s, vt = lg.svds(CI_1_sq, k = k_svd)

sigma = np.diagflat(s)  # Form a diagonal matrix from vector S
CI_1_sq = np.dot(np.dot(u,sigma),vt)
normilization=1/np.linalg.norm(CI_1_sq)
CI_1_sq=CI_1_sq*normilization
#print(CI_1_sq)
print(np.linalg.norm(CI_1_sq))


# In[28]:


CiA=CI_1_sq
#print(type(CiA))
CiA=CiA/np.linalg.norm(CiA)
CiA = CiA.ravel()
ci=CiA.tolist()


#print(ci)
#print(np.dot(CiA,CiA),"Normilization Check")




import struct
import numpy
f=open("recn_init.bin","wb")
bin=struct.pack('d'*len(ci),*ci)
f.write(bin)
f.close()



# In[ ]:




