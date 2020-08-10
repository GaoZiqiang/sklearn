
# coding: utf-8

# In[2]:


from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


faces = fetch_lfw_people(min_faces_per_person=60)

faces.images.shape


# In[ ]:


faces.data.shape
X = faces.data

