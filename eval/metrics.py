import numpy as np
def reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    error=np.sqrt(np.sum((original-reconstructed)**2,axis=1))
    return error
# ithe aapn original jo array hota data cha ani jo reconstuct houn aala ahe tyamadeh kiti diffrence ahe to forbenius error chya rupat kadat ahe ithe axis 1 vaparla ahe karan aaplyala each row wise kiti error ala ahe te calculate karycha ahe