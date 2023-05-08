import torch as th

def masked_softmax(x: th.Tensor, mask: th.Tensor, beta: th.Tensor=th.Tensor([1.0]), dim:int=0) -> th.Tensor:
    sc = beta * x
    sc.masked_fill_(mask, -1E20)
    return th.nn.functional.softmax(sc, dim=dim)

def entropy(x: th.Tensor):
    i = th.log(x)*x
    i[i!=i] = 0.0
    return -i.sum().item()