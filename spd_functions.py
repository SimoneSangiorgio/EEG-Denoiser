import torch

# X: SPD matrix
# u: matrix whose column vectors are the eigenvectors of X
# s: vector containing the eigenvalues of X

def tensor_power(X, scalar):
    s, u = torch.linalg.eigh(X)
    return torch.matmul(torch.matmul(u,torch.diag_embed(s**scalar)),u.transpose(1,2))

def spd_dis(X1,X2):
    matrix_X1X2X1 = torch.matmul(torch.matmul(tensor_power(X1,-0.5),X2),tensor_power(X1,-0.5))
    s,u = torch.linalg.eigh(matrix_X1X2X1)
    s_trans = (torch.log(s))**2
    return s_trans.sum(1)

def spd_logm(X):
    s, u = torch.linalg.eigh(X)
    return torch.matmul(torch.matmul(u,torch.diag_embed(torch.log(s))),u.transpose(1,2))

def spd_expm(X):
    s, u = torch.linalg.eigh(X)
    return torch.matmul(torch.matmul(u,torch.diag_embed(torch.exp(s))),u.transpose(1,2))

def spd_mul(X, scalar):
    return spd_expm(scalar * spd_logm(X))

def spd_plus(X1, X2):
    return spd_expm(spd_logm(X1) + spd_logm(X2))

def spd_minus(X1, X2):
    return spd_expm(spd_logm(X1) - spd_logm(X2))

def spd_verificator(X):
    is_symmetric = torch.allclose(X, X.transpose(-1, -2), atol=1e-6)
    print(f"Symmetric matrix? {is_symmetric}")
    eigenvalues, _ = torch.linalg.eigh(X)
    is_spd = torch.all(eigenvalues > 0)
    print(f"SPD matrix? {is_spd}")
    #print(f"Gli autovalori sono: {torch.linalg.eigh(X)[0]}") 

def spd_snr(C_original, C_denoised):

    assert C_original.shape == C_denoised.shape, "same dimensions matrices"
    
    signal_power = torch.norm(C_original, 'fro')**2
    noise_power = torch.norm(C_original - C_denoised, 'fro')**2
    if noise_power == 0:
        return float('inf')
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()