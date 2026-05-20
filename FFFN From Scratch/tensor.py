import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data)
        self._prev = set(_children)
        self._op = _op
        self.grad = np.zeros_like(data)
        self._backward = lambda: None
        
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad  += self._unbroadcast(out.grad)
            other.grad += self._unbroadcast(out.grad)
        out._backward = _backward

        return out

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad  += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out
    
    def apply(self, func, grad_func):
        out = Tensor(func(self.data), (self,), 'activation')

        def _backward():
            self.grad += out.grad * grad_func(self.data)

        out._backward = _backward
        return out
    
    def _backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.grad)
        for t in reversed(topo):
            t._backward()

    def _unbroadcast(self, grad):
        ndims_added = grad.ndim - len(self.data.shape)
        if ndims_added > 0:
            grad = grad.sum(axis=tuple(range(ndims_added)))
        axes = tuple(i for i, (d_out, d_orig) in enumerate(zip(grad.shape, self.data.shape)) if d_orig == 1 and d_out > 1)
        if axes:
            grad = grad.sum(axis=axes, keepdims=True)
        return grad