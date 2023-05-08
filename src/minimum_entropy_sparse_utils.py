import numpy as np
import torch as th
from torch.utils.cpp_extension import load
lltm_cpp = load(name="lltm_cpp", sources=["lltm.cpp"])
lltm_lemma3 = lltm_cpp.lemma3

def greatest_lower_bound(p, q):  # compute p hat q
    # https://arxiv.org/pdf/1901.07530.pdf (fact 1)
    # input: p, q of dims batch x dim
    assert p.shape == q.shape, "p, q need to have same dimensions!"
    dim = p.shape[1]
    z = p.clone().zero_()
    z[:, 0] = th.stack([th.minimum(q[b, 0], p[b, 0]) for b in range(q.shape[0])], -1)
    cumsum_p = th.cumsum(p, -1)
    cumsum_q = th.cumsum(q, -1)
    for i in range(1, dim):
        sum_z = th.sum(z[:, :i], -1)
        z[:, i] = th.stack([th.minimum(cumsum_p[b, i], cumsum_q[b, i]) for b in range(q.shape[0])], -1) - sum_z
    return z


def greatest_lower_bound_fast(p, q):  # compute p hat q
    # https://arxiv.org/pdf/1901.07530.pdf (fact 1)
    # input: p, q of dims batch x dim
    assert p.shape == q.shape, "p, q need to have same dimensions!"
    dim = p.shape[1]
    z = p.clone().zero_()
    z[:, 0] = th.stack([th.minimum(q[b, 0], p[b, 0]) for b in range(q.shape[0])], -1)
    cumsum_p = th.cumsum(p, -1)
    cumsum_q = th.cumsum(q, -1)
    for i in range(1, dim):
        sum_z = th.sum(z[:, :i], -1)
        z[:, i] = th.stack([th.minimum(cumsum_p[b, i], cumsum_q[b, i]) for b in range(q.shape[0])], -1) - sum_z
    return z

def Lemma3(z, x, A, i, j):
    # assert z, x, A all properly behaved with i, j being explicit indices
    #assert not z[z<=0.0].sum(), "ERROR: z needs to larger than zero!"
    #assert not x[x < 0.0].sum(), "ERROR: x needs to be non-negative!"
    #assert not A[A < 0.0].sum(), "ERROR: x needs to be non-negative!"
    assert x >= 0, "ERROR: x needs to be non-negative!"
    assert th.isclose(A[A < 0.0].sum(), th.tensor([0.0], dtype=th.float64)), "ERROR: A needs to be non-negative!"
    assert not( A.sum() + x < z and not th.isclose(A.sum() + x, z)), "ERROR: A, x, z do not fulfill A.sum() + x >= z!"

    # if (th.isclose(A[k], th.tensor([0.0], dtype=th.float64)) and th.isclose(x, th.tensor([0.0], dtype=th.float64)))
    # or th.isclose(x, th.tensor([0.0], dtype=th.float64))
    k = i
    I = set()
    sum = 0.0
    # NOTE: paper does not treat special case where both x AND A[k] are zero (or close to zero actually)
    while True: # or th.isclose(th.tensor([sum + A[k].item()], dtype=th.float64), x)):
        #sum = sum + A[k].item()
        if (A[:k].sum() > x):
            break
        I.add(k)
        k = k + 1
        if k > j:
            break

    # I.add(k) # Note: This is a BUG in the paper!
    #z_d = x - sum
    #z_r = z - z_d
    z_r = A[:k].sum() - x
    z_d = z - z_r #A[k-1] - z_r
    return z_d, z_r, I#, k-1


# TODO: Implement Lemma 3 here
def minimum_entropy_coupling(p, q, algo_atol=1E-8, warning_atol=1E-8, rtol=None, mode="dense", verbose=False):
    # https://arxiv.org/pdf/1901.07530.pdf (Algorithm 1)
    # input: p, q are 1D pytorch tensors describing probability distributions
    assert p.dim()==1 and q.dim()==1, "ERROR: batch mode not yet supported!"

    # assert p, q are distributions
    assert th.isclose(p.sum(), th.tensor([1.0], dtype=p.dtype)) and \
                      th.isclose(th.sum(p[p<0.0]), th.tensor([0.0], dtype=p.dtype)) and \
                                 th.isclose(q.sum(), th.tensor([1.0], dtype=p.dtype)) and \
                                            th.isclose(th.sum(q[q<0.0]), th.tensor([0.0], dtype=p.dtype)), \
                                 "Either q, p (or both) are not proper probability distributions! p: {} q: {}".format(p, q)

    p_orig = p.clone()
    q_orig = q.clone()

    # put p, q in non-increasing order
    p, p_indices = p.sort(descending=True)
    q, q_indices = q.sort(descending=True)

    # pad input distributions if necessary
    q_pad = None
    p_pad = None
    if p.shape[0] > q.shape[0]:
        q_pad = p.shape[0] - q.shape[0]
        q = th.cat([q, th.zeros(q_pad)], -1)
    elif p.shape[0] < q.shape[0]:
        p_pad = q.shape[0] - p.shape[0]
        p = th.cat([p, th.zeros(p_pad)], -1)
    n = p.shape[0]
    M = th.zeros((n,n), dtype=th.float64)

    # should we swap p, q?
    if (p-q).sum() > 0.0:
        i = -1
        for j in range(n):
            if p[j] != p[i]:
                i = j
        # swap p, q
        if p[i] < p[j]:
            p, q = q.clone(), p.clone() # swap variables for some reason (NOTE: Do we not have to undo this swapping in the end?)

    # compute greatest lower bound
    #z = greatest_lower_bound(p.unsqueeze(0), q.unsqueeze(0)).squeeze(0)
    #z = lltm_cpp.glb(p.unsqueeze(0), q.unsqueeze(0)).squeeze(0)
    #z = greatest_lower_bound_fast(p.unsqueeze(0), q.unsqueeze(0)).squeeze(0)
    z = lltm_cpp.glbfast(p.unsqueeze(0), q.unsqueeze(0)).squeeze(0)
    for i in range(n):
        M[i,i] = z[i]

    if mode in ["dense"]:
        M = lltm_cpp.mec(n, M, q, p, z, verbose, algo_atol)
    elif mode in ["sparse"]:
        M = lltm_cpp.mec_sparse(p.clone(), q.clone(), z.clone(), verbose, algo_atol).clone()  # (n, Mp, q, p, z)
    else:
        raise Exception("MEC: Unknown sparsity mode: {}".format(mode))

    # Return M into sorted order
    if p_pad is not None:
        M = M[:-p_pad, :]
    if q_pad is not None:
        M = M[:, :-q_pad]
    M = M[p_indices.argsort(), :]
    M = M[:, q_indices.argsort()]

    # CHECK whether marginals are correct!
    p_error = 0.0
    q_error = 0.0
    is_warnings = False
    if not th.isclose(M.sum(dim=1), p_orig, atol=warning_atol).all():
        th.set_printoptions(precision=10)
        p_error = th.max(th.abs(M.sum(1)-p_orig)).item()
        if (verbose):
            print("P marginal is incorrect! error: {}  p: {} p_marg: {} q: {}".format(p_error, p_orig, M.sum(1), q))
        is_warnings = True
    if not th.isclose(M.sum(dim=0), q_orig, atol=warning_atol).all():
        th.set_printoptions(precision=10)
        q_error = th.max(th.abs(M.sum(0)-q_orig)).item()
        if(verbose):
            print("Q marginal is incorrect! error: {} q: {} q_marg: {} p: {}".format(q_error, q_orig, M.sum(0), p))
        is_warnings = True

    best_entropy = -th.sum(z[z>=algo_atol]*th.log2(z[z>=algo_atol])).item()
    M_entropy = -th.sum(M[M>=algo_atol]*th.log2(M[M>=algo_atol])).item()

    if not (M_entropy - best_entropy <= 1.0):
        print("Additive Gap larger than guaranteed (1 bit max): {} for M: {}".format(M_entropy - best_entropy, M))
    return {"M": M,
            "best_entropy": best_entropy,
            "M_entropy": M_entropy,
            "additive_gap": M_entropy - best_entropy,
            "warnings": is_warnings,
            "q_error": q_error,
            "p_error": p_error}

if __name__ == "__main__":
    # Unit test 1 - greatest lower bound
    p = th.from_numpy(np.array([[0.4, 0.3, 0.15, 0.08, 0.04, 0.03]], dtype=np.float64))
    q = th.from_numpy(np.array([[0.44, 0.18, 0.18, 0.15, 0.03, 0.02]], dtype=np.float64))
    p /= p.sum()
    q /= q.sum()
    assert np.isclose(greatest_lower_bound(p, q).cpu().numpy(), np.array([0.4, 0.22, 0.18, 0.13, 0.04, 0.03], dtype=np.float64)).all(), "ERROR in greatest lower bound routine!"
    print("Test 1 passed.")

    # Unit test 2 - minimum entropy coupling
    M = minimum_entropy_coupling(p.squeeze(0), q.squeeze(0))
    # NOTE: This is not exactly how the example in the paper worked!
    M_true = np.array([[0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.04, 0.18, 0.03, 0.05, 0.0, 0.0],
                       [0.0, 0.0, 0.15, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.08, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.02, 0.02, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.01, 0.02]], dtype=np.float64)
    print("M p marg:", M["M"].sum(0))
    print("M q marg:", M["M"].sum(1))
    assert np.isclose(M["M"].sum(0).numpy(),q).all(), "ERROR: q not marginalised properly!"
    assert np.isclose(M["M"].sum(1).numpy(),p).all(), "ERROR: p not marginalised properly!"

    entropy_paper = -th.sum(th.log2(th.from_numpy(M_true[M_true!=0.0]))*th.from_numpy(M_true[M_true!=0.0]))
    entropy_our = -th.sum(th.log2(M["M"][M["M"]!=0.0]) * M["M"][M["M"]!=0.0])
    print("Entropy our: {}, paper: {}".format(entropy_our, entropy_paper))
    print("Test 2 passed.")

    # TEST 3: Some other instability encountered!
    p = th.DoubleTensor([1., 0.])
    q = th.DoubleTensor([0.0000, 0.8364, 0.0599, 0.1037])
    M = minimum_entropy_coupling(p,q)["M"]
    assert np.isclose(M.sum(0).numpy(),q).all(), "ERROR: q not marginalised properly!"
    assert np.isclose(M.sum(1).numpy(),p).all(), "ERROR: p not marginalised properly!"
    print("Test 3 passed.")

    p = th.from_numpy(np.array([[0.5535, 0.4465]], dtype=np.float64))
    q = th.from_numpy(np.array([[0.3436, 0.3156, 0.1729, 0.1678]], dtype=np.float64))
    p /= p.sum()
    q /= q.sum()
    M = minimum_entropy_coupling(p.squeeze(0), q.squeeze(0))["M"]
    print("Marginal p: {} (true: {})".format(M.sum(dim=1), p))
    print("Marginal q: {} (true: {})".format(M.sum(dim=0), q))
    assert np.isclose(M.sum(0).numpy(),q).all(), "ERROR: q not marginalised properly!"
    assert np.isclose(M.sum(1).numpy(),p).all(), "ERROR: p not marginalised properly!"
    print("Test 4 passed.")

    p = th.from_numpy(np.array([0.1194241691, 0.0492131407, 0.1149148347, 0.1194241688, 0.1194241689,
        0.1194241680, 0.0000000000, 0.1194133982, 0.1193377967, 0.1194241550]))
    q = th.from_numpy(np.array([9.9990618229e-01, 9.3856368039e-05, 1.2170937147e-08, 6.5520471806e-15,
        5.1764567804e-15, 6.3972500676e-20, 0.0000000000e+00, 0.0000000000e+00,
        0.0000000000e+00, 0.0000000000e+00]))
    p /= p.sum()
    q /= q.sum()
    M = minimum_entropy_coupling(p.squeeze(0), q.squeeze(0))["M"]
    print("Marginal p: {} (true: {})".format(M.sum(dim=1), p))
    print("Marginal q: {} (true: {})".format(M.sum(dim=0), q))
    assert np.isclose(M.sum(0).numpy(),q).all(), "ERROR: q not marginalised properly!"
    assert np.isclose(M.sum(1).numpy(),p).all(), "ERROR: p not marginalised properly!"
    print("Test 5 passed.")

    p = th.from_numpy(np.array([0.1793738642, 0.1031307858, 0.0000000000, 0.1793738642, 0.1793738642,
        0.0000000000, 0.0000000000, 0.1793738642, 0.0000000000, 0.1793737573]))
    q = th.from_numpy(np.array([0.4220781326, 0.4198269546, 0.1580949277, 0.0000000000, 0.0000000000,
        0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000]))
    p /= p.sum()
    q /= q.sum()
    M = minimum_entropy_coupling(p.squeeze(0), q.squeeze(0))["M"]
    print("Marginal p: {} (true: {})".format(M.sum(dim=1), p))
    print("Marginal q: {} (true: {})".format(M.sum(dim=0), q))
    assert np.isclose(M.sum(0).numpy(),q).all(), "ERROR: q not marginalised properly!"
    assert np.isclose(M.sum(1).numpy(),p).all(), "ERROR: p not marginalised properly!"
    print("Test 6 passed.")



    p = th.DoubleTensor([0.1194241691, 0.0492131407, 0.1149148347, 0.1194241688, 0.1194241689,
     0.1194241680, 0.0000000000, 0.1194133982, 0.1193377967, 0.1194241550])
    q = th.DoubleTensor([6.5520471806e-15, 9.9990618229e-01, 5.1764567804e-15, 9.3856368039e-05,
     6.3972500676e-20, 1.2170937147e-08])
    p /= p.sum()
    q /= q.sum()
    M = minimum_entropy_coupling(p.squeeze(0), q.squeeze(0))["M"]
    print("Marginal p: {} (true: {})".format(M.sum(dim=1), p))
    print("Marginal q: {} (true: {})".format(M.sum(dim=0), q))
    assert np.isclose(M.sum(0).numpy(),q).all(), "ERROR: q not marginalised properly!"
    assert np.isclose(M.sum(1).numpy(),p).all(), "ERROR: p not marginalised properly!"
    print("Test 7 passed.")

    p = th.DoubleTensor([0.1194241691, 0.0492131407, 0.1149148347, 0.1194241688, 0.1194241689,
        0.1194241680, 0.0000000000, 0.1194133982, 0.1193377967, 0.1194241550])
    q = th.DoubleTensor([6.5520471806e-15, 9.9990618229e-01, 5.1764567804e-15, 9.3856368039e-05,
        6.3972500676e-20, 1.2170937147e-08])
    p /= p.sum()
    q /= q.sum()
    M = minimum_entropy_coupling(p.squeeze(0), q.squeeze(0))["M"]
    print("Marginal p: {} (true: {})".format(M.sum(dim=1), p))
    print("Marginal q: {} (true: {})".format(M.sum(dim=0), q))
    assert np.isclose(M.sum(0).numpy(),q).all(), "ERROR: q not marginalised properly!"
    assert np.isclose(M.sum(1).numpy(),p).all(), "ERROR: p not marginalised properly!"
    print("Test 8 passed.")

    pass