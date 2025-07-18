import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool

from e3nn.o3 import Irreps

class InstanceNorm(nn.Module):
    '''Instance normalization for orthonormal representations
    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.
    Parameters
    ----------
    irreps : `Irreps`
        representation
    eps : float
        avoid division by zero when we normalize by the variance
    affine : bool
        do we have weight and bias parameters
    reduce : {'mean', 'max'}
        method used to reduce
    '''

    def __init__(self, irreps, eps=1e-5, affine=True, reduce='mean', normalization='component'):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        # counts the number of 0e or 0o irreps
        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0)
        # total number of irreps
        num_features = self.irreps.num_irreps

        if affine:

            self.weight = nn.Parameter(torch.ones(num_features))
            # adds bias to scalars only
            self.bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ['mean', 'max'], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    # special method used to define a human-readable string representation of an object.
    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    def forward(self, input, batch):
        '''evaluate
        Parameters
        ----------
        input : `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        '''
        # batch, *size, dim = input.shape  # TODO: deal with batch
        # input = input.reshape(batch, -1, dim)  # [batch, sample, stacked features]
        # input has shape [batch * nodes, dim], but with variable nr of nodes.
        # the input batch slices this into separate graphs
        dim = input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            field = input[:, ix: ix + mul * d]  # [batch * sample, mul * repr]
            ix += mul * d

            # [batch * sample, mul, repr]
            field = field.reshape(-1, mul, d)

            # For scalars first compute and subtract the mean
            if ir.l == 0:
                # Compute the mean
                # print(f"field: {field}")
                # print(f"batch: {batch}")
                # print(f"mul: {mul}")
                field_mean = global_mean_pool(field, batch).reshape(-1, mul, 1)  # [batch, mul, 1]]
                # Subtract the mean
                field = field - field_mean[batch]

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError("Invalid normalization option {}".format(self.normalization))
            # Reduction method
            if self.reduce == 'mean':
                field_norm = global_mean_pool(field_norm, batch)  # [batch, mul]
            elif self.reduce == 'max':
                field_norm = global_max_pool(field_norm, batch)  # [batch, mul]
            else:
                raise ValueError("Invalid reduce option {}".format(self.reduce))

            # Then apply the rescaling (divide by the sqrt of the squared_norm, i.e., divide by the norm
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.weight[None, iw: iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm[batch].reshape(-1, mul, 1)  # [batch * sample, mul, repr]

            if self.affine and d == 1:  # scalars
                bias = self.bias[ib: ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output