import numpy as np

X = [[3,4],[0,4],[0,1]] 
# (n = 3, d = 2)

mean=np.mean(X)
print(mean)
Xᶜ=X-mean
eigenvalues, eigenvectors = np.linalg.eig(Xᶜ)
print(eigenvalues,eigenvectors)
# (b) C काढ आणि त्याचे eigenvalues/top eigenvector (दिशा) काढ.=