import numpy as np

# T13 Stability Test: Covariance पद्धत आणि Direct SVD पद्धतीमधील Numerical Stability ची तुलना करण्यासाठी टेबल हेडर
print('\n--- T13 Stability Table ---')
print('| kappa | cond(Xc) | max|mean| | true sigma_min | cov sigma_min | svd sigma_min | err_cov | err_svd | cov >= svd | cov >= 100*svd |')
print('|---|---|---|---|---|---|---|---|---|---|')

# वेगवेगळ्या Condition Numbers (kappa) ची लिस्ट (लहान 100 पासून ते अतिशय अस्थिर/Ill-conditioned 10^8 पर्यंत)
kappas = [1e2, 1e4, 1e6, 1e7, 1e8]
n, d = 100, 5 # n = 100 सॅम्पल्स (rows), d = 5 फीचर्स (columns)

for kappa in kappas:
    # Singular values (sigma): पहिली = 1.0 (मॅक्सिमम), शेवटची = 1.0/kappa (किमान)
    # यामुळे मॅट्रिक्सचा अचूक Condition Number = 1.0 / (1.0/kappa) = kappa तयार होतो
    s = np.array([1.0, 0.5, 0.2, 0.1, 1.0/kappa])
    
    # Reproducible (पुन्हा तेच रिझल्ट येण्यासाठी) Random Generator
    rng = np.random.default_rng(42)
    
    # Zero-mean आणि Orthogonal मॅट्रिक्स U तयार करण्यासाठी QR decomposition
    U_raw = rng.normal(size=(n, d))
    U_raw -= np.mean(U_raw, axis=0)
    U, _ = np.linalg.qr(U_raw)
    U -= np.mean(U, axis=0)
    U, _ = np.linalg.qr(U)
    
    # d x d चा Orthogonal मॅट्रिक्स V (Right singular vectors) तयार करतो
    V, _ = np.linalg.qr(rng.normal(size=(d, d)))
    
    # SVD फॉर्म्युला (X = U @ Sigma @ V.T) वापरून ठरवलेल्या kappa चा सिंथेटिक डेटा X बनवला
    X = U @ np.diag(s) @ V.T
    
    # कॉलम-वाईज मीन आणि कमाल मीन तपासून डेटा सेंटर्ड आहे का ते पाहणे
    mean_X = np.mean(X, axis=0)
    max_mean = np.max(np.abs(mean_X))
    
    # डेटा झिरो-सेंटर करून प्रत्यक्ष डेटाचा Condition Number cond(Xc) मोजणे
    Xc = X - np.mean(X, axis=0)
    cond_Xc = np.linalg.cond(Xc)
    
    # मॅट्रिक्सची खरी (True) सर्वात लहान Singular Value
    sigma_min_true = 1.0 / kappa
    
    # --- पद्धत १: Covariance Method (X.T @ X) ---
    # यात Condition Number चा वर्ग (kappa^2) होतो, ज्यामुळे फ्लोटिंग-पॉईंट एरर वाढतो
    C = X.T @ X / (n - 1)
    evals, _ = np.linalg.eigh(C)
    sigma_min_cov = np.sqrt(max(0, evals[0] * (n - 1)))
    
    # --- पद्धत २: Direct SVD Method (np.linalg.svd) ---
    # यात डेटाचा वर्ग होत नाही, त्यामुळे ही पद्धत जास्त स्थिर (Numerically Stable) राहते
    _, S_svd, _ = np.linalg.svd(X)
    sigma_min_svd = S_svd[-1]
    
    # दोन्ही पद्धतींचा खऱ्या व्हॅल्यूशी Relative Error काढणे
    err_cov = np.abs(sigma_min_cov - sigma_min_true) / sigma_min_true
    err_svd = np.abs(sigma_min_svd - sigma_min_true) / sigma_min_true
    
    # kappa = 1e8 असताना Covariance चा एरर SVD पेक्षा १०० पटीने जास्त आहे का (cov >= 100*svd) हे तपासणे
    last_col = (err_cov >= 100*err_svd) if kappa == 1e8 else "-"
    
    # सर्व आकडे टेबल फॉरमॅटमध्ये स्क्रीनवर दाखवणे
    print(f'| {kappa} | {cond_Xc:.6g} | {max_mean:.6g} | {sigma_min_true:.6g} | {sigma_min_cov:.6g} | {sigma_min_svd:.6g} | {err_cov:.6g} | {err_svd:.6g} | {err_cov >= err_svd} | {last_col} |')
