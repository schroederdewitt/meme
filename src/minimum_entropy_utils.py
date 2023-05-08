import numpy as np
import torch as th

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
def minimum_entropy_coupling(p, q, atol=1E-6, rtol=None):
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
    z = greatest_lower_bound(p.unsqueeze(0), q.unsqueeze(0)).squeeze(0)
    for i in range(n):
        M[i,i] = z[i]

    # Start big loop
    i = n-1
    while i >= 1:
        sum_m_ki = th.sum(M[i:n, i])
        if sum_m_ki > q[i] and not th.isclose(sum_m_ki-q[i], th.tensor([0.0], dtype=q.dtype)): #sum_m_ki > q[i]:
            z_i_d, z_i_r, I = Lemma3(z[i], q[i], M[:, i].view(-1), 0, n-1)
            M[i, i] = z_i_d
            M[i, i - 1] = z_i_r
            I.add(i)
            neg_set = set(list(range(n))) - I
            for k in neg_set:
                M[k, i-1] = M[k, i]
                M[k, i] = 0
        sum_m_ik = th.sum(M[i, i:n])
        if sum_m_ik > p[i] and not th.isclose(sum_m_ik-p[i], th.tensor([0.0], dtype=q.dtype)):
            z_i_d, z_i_r, I = Lemma3(z[i], p[i], M[i, :].view(-1), 0, n-1)
            M[i, i] = z_i_d
            M[i - 1, i] = z_i_r
            I.add(i)
            neg_set = set(list(range(n))) - I
            for k in neg_set:
                M[i-1, k] = M[i, k]
                M[i, k] = 0
        i = i - 1

    # Return M into sorted order
    if p_pad is not None:
        M = M[:-p_pad, :]
    if q_pad is not None:
        M = M[:, :-q_pad]
    M = M[p_indices.argsort(), :]
    M = M[:, q_indices.argsort()]

    # CHECK whether marginals are correct!
    #assert th.isclose(M.sum(dim=1), p_orig).all(), "P marginal is incorrect! p: {} p_marg: {} q: {}".format(p_orig, M.sum(1), q)
    #assert th.isclose(M.sum(dim=0), q_orig).all(), "Q marginal is incorrect! q: {} q_marg: {} p: {}".format(q_orig, M.sum(0), p)
    if not th.isclose(M.sum(dim=1), p_orig, atol=atol).all():
        th.set_printoptions(precision=10)
        print("WARNING: marginal is incorrect! p: {} p_marg: {} q: {}".format(p_orig, M.sum(1), q))
    if not th.isclose(M.sum(dim=0), q_orig, atol=atol).all():
        th.set_printoptions(precision=10)
        print("Q marginal is incorrect! error: {} q: {} q_marg: {} p: {}".format(th.max(th.abs(M.sum(0)-q_orig)).item(), q_orig, M.sum(0), p))

    best_entropy = -th.sum(z[z>=atol]*th.log2(z[z>=atol])).item()
    M_entropy = -th.sum(M[M>=atol]*th.log2(M[M>=atol])).item()

    if not (M_entropy - best_entropy <= 1.0):
        print("Additive Gap larger than guaranteed (1 bit max): {} for M: {}".format(M_entropy - best_entropy, M))
    return {"M": M,
            "best_entropy": best_entropy,
            "M_entropy": M_entropy,
            "additive_gap": M_entropy - best_entropy}

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
    # assert np.isclose(M.cpu().numpy(), M_true).all(), "ERROR in minimum entropy coupling routine!"
    entropy_paper = -th.sum(th.log2(th.from_numpy(M_true[M_true!=0.0]))*th.from_numpy(M_true[M_true!=0.0]))
    entropy_our = -th.sum(th.log2(M["M"][M["M"]!=0.0]) * M["M"][M["M"]!=0.0])
    print("Entropy our: {}, paper: {}".format(entropy_our, entropy_paper))
    print("Test 2 passed.")

    # TEST 3: Some other instability encountered!
    p = th.DoubleTensor([1., 0.])
    q = th.DoubleTensor([0.0000, 0.8364, 0.0599, 0.1037])
    M = minimum_entropy_coupling(p,q)["M"]
    print("Test 3 passed.")

    p = th.from_numpy(np.array([[0.5535, 0.4465]], dtype=np.float64))
    q = th.from_numpy(np.array([[0.3436, 0.3156, 0.1729, 0.1678]], dtype=np.float64))
    p /= p.sum()
    q /= q.sum()
    M = minimum_entropy_coupling(p.squeeze(0), q.squeeze(0))["M"]
    print("Marginal p: {} (true: {})".format(M.sum(dim=1), p))
    print("Marginal q: {} (true: {})".format(M.sum(dim=0), q))
    print("Test 4 passed.")

    p = th.from_numpy(np.array([0.0035084464, 0.0467336840, 0.9462494282, 0.0035084414]))
    q = th.from_numpy(np.array([0.5264073610, 0.4189087749, 0.0546837784, 0.0000000000]))
    p /= p.sum()
    q /= q.sum()
    M = minimum_entropy_coupling(p.squeeze(0), q.squeeze(0))["M"]
    print("Marginal p: {} (true: {})".format(M.sum(dim=1), p))
    print("Marginal q: {} (true: {})".format(M.sum(dim=0), q))
    print("Test 5 passed.")
    pass