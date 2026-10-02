import numpy as np
from error import InputValidation


def solve_linear_system(a, b):
    answer = np.linalg.solve(a=a, b=b)
    return answer


# ithe aapn gaussian elimation method use karun x ani y chya values find karat aahot


def power_iteration(A: np.ndarray, tol: int, max_iter: int, seed):
    if not np.allclose(A, A.T):
        raise InputValidation(message="A is wrong")
    A_shape = A.shape
    rng = np.random.default_rng(seed)
    v0 = rng.random(A_shape[0])

    mul = 0
    converged = False

    for i in range(max_iter):
        mul = A @ v0
        normalize = np.linalg.norm(mul)
        if np.allclose(normalize, 0):
            raise InputValidation(message="Cannot normalize zero vector")
        v = mul / normalize
        convergence = v0 - v
        convergence_norm = np.linalg.norm(convergence)
        if convergence_norm < tol:
            converged = True
            break
        v0 = v
        
    if not converged:
        raise InputValidation(message="Power iteration did not converge")
    eigenvalue = (v.T @ A @ v) / (v.T @ v)
    eigenvector = v
    iterations_used = i + 1
    return {
        "eigenvalue": eigenvalue,
        "eigenvector": eigenvector,
        "iterations_used": iterations_used,
    }


# aaplyaya power iteration function mmahde aapn ek A value ghetlu ahe tol max iter mhanje maximim kiti loop firla pahije ani seed adi aapn check kela ahe ki jr not np.allclose(A, A.T) mhanjech (9,9) ashech shape allow aahet karan jr (1,32) kela tr tycha inverse barobar yet nahi mhanun te nahi khali aan rng set keli ahe tya madhe aaplyala each column madhe same random number set honar ahet khali looop laun
# aapn adi multiplication kela ahe A ani a0 sobat parat normalize ani
#  he ka check kela ahe tr
# convergence_norm = np.linalg.norm(convergence)if convergence_norm < tol:    break


# याचा अर्थ:
# - convergence = v0 - v → old आणि new vector मधला फरक
# - np.linalg.norm(convergence) → त्या फरकाची size/magnitude
# - tol = आपण किती छोटा फरक acceptable मानणार ते
# जर:
# convergence_norm < tol

# तर:
# new vector आणि old vector जवळजवळ same आहेत → algorithm converge झाला → पुढे loop चालवण्याची गरज नाही.=convergence_norm=np.linalg.norm(convergence)
# if convergence_norm<tol:
# break tr jr
#  last la eigen value kadlya aahet manually iwthout using any method


def deflate(A, eigenvalue, eigenvector: np.ndarray):
    v = eigenvector.reshape(
        -1, 1
    )  ## ithe aapla eigenvector je aahet te 1 dimension shape made yeu shaktat eg(2,) pn khali muiltiply karnyasathi aapn tyala 2 dimension madhe convert karto
    mul = (
        v @ v.T
    )  ##इथे eigenvalue नाही, तर eigenvector पासून matrix/component तयार केला जातो.
    mul = eigenvalue * mul  ##dominant eigen-component मिळतो.
    deflated = (
        A - mul
    )  ## ani ithe आपण specifically पहिल्या/dominant eigenpair शी संबंधित component subtract करतो. so second eigenvalue kadyla yeil
    return deflated  ##ani substracted information cha array return kela


# deflate function madhe aaplyala remaining eigenvalue find karychya ahet first strongest eginevalue hetlya ahet ata reaminig n 2 tyachya khalchya find karychya aahet


def top_k_eigenpairs(A, k, tol, max_iter, seed):
    eigenpairs = []
    current_A = A.copy()

    for i in range(k):
        eigenv_vector_values = power_iteration(
            A=current_A, tol=tol, max_iter=max_iter, seed=seed
        )

        eigenpairs.append(
            {
                "eigenvalue": eigenv_vector_values["eigenvalue"],
                "eigenvector": eigenv_vector_values["eigenvector"],
            }
        )

        deflate_output = deflate(
            A=current_A,
            eigenvalue=eigenv_vector_values["eigenvalue"],
            eigenvector=eigenv_vector_values["eigenvector"],
        )

        current_A = deflate_output
    sorted_pairs = sorted(eigenpairs, key=lambda x: abs(x["eigenvalue"]), reverse=True)
    return sorted_pairs
