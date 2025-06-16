import torch
from torch import nn
from torch.autograd import Function
from spd_functions import *

class StiefelParameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)
    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()

class SPDIncreaseDim(nn.Module):

    def __init__(self, input_size, output_size):
        super(SPDIncreaseDim, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.register_buffer('eye', torch.eye(output_size, input_size).to(self.device))
        add = torch.as_tensor([0] * input_size + [1] * (output_size-input_size), dtype=torch.float64)
        add = add.to(self.device)
        self.register_buffer('add', torch.diag(add))

    def forward(self, input):
        eye = self.eye.unsqueeze(0)
        eye = eye.expand(input.size(0), -1, -1).double()
        add = self.add.unsqueeze(0)
        add = add.expand(input.size(0), -1, -1)

        output = torch.baddbmm(add, eye, torch.bmm(input, eye.transpose(1,2)))

        return output

class SPDTransform(nn.Module):

    def __init__(self, input_size, output_size, time_dim):
        super(SPDTransform, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.increase_dim = None
        if output_size > input_size:
            self.increase_dim = SPDIncreaseDim(input_size, output_size)
            input_size = output_size
        self.weight = StiefelParameter(torch.DoubleTensor(input_size, output_size).to(self.device), requires_grad=True)
        nn.init.orthogonal_(self.weight)

        self.emb_layer = nn.Sequential( 
            nn.Linear(time_dim,time_dim).double(),
            nn.SiLU(), 
            nn.Linear(time_dim,output_size*output_size).double(),)        

    def forward(self, input, t):

        output = input
        emb = self.emb_layer(t)

        if self.increase_dim:
            output = self.increase_dim(output)

        weight = self.weight.unsqueeze(0).expand(input.size(0), -1, -1)
        output = torch.bmm(weight.transpose(1,2), torch.bmm(output, weight))

        t_emb = emb.reshape(output.shape[0] ,output.shape[1], output.shape[1])

        output = torch.matmul(torch.matmul(t_emb,output),t_emb.transpose(1,2))

        return output

def symmetric(A):
    size = list(range(len(A.shape)))
    temp = size[-1]
    size.pop()
    size.insert(-1, temp)
    return 0.5 * (A + A.permute(*size))

class SPDRectifiedFunction(Function):

    @staticmethod
    def forward(ctx, input, epsilon):
        ctx.save_for_backward(input, epsilon)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s[s < epsilon[0]] = epsilon[0]

            output[k] = u.mm(s.diag().mm(u.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, epsilon = ctx.saved_tensors
        grad_input = None
        
        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1); eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(2))
            for k, g in enumerate(grad_output):
                if len(g.shape) == 1:
                    continue

                g = symmetric(g)

                x = input[k]
                u, s, v = x.svd()
                
                max_mask = s > epsilon
                s_max_diag = s.clone(); s_max_diag[~max_mask] = epsilon; s_max_diag = s_max_diag.diag()
                Q = max_mask.float().diag().double()
                
                dLdV = 2*(g.mm(u.mm(s_max_diag)))
                dLdS = eye * (Q.mm(u.t().mm(g.mm(u))))
                
                P = s.unsqueeze(1)
                P = P.expand(-1, P.size(0))
                P = P - P.t()
                mask_zero = torch.abs(P) == 0
                P = 1 / P
                P[mask_zero] = 0

                grad_input[k] = u.mm(symmetric(P.t() * u.t().mm(dLdV))+dLdS).mm(u.t())
            
        return grad_input, None

class SPDRectified(nn.Module):

    def __init__(self, epsilon=1e-4):
        super(SPDRectified, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.register_buffer('epsilon', torch.DoubleTensor([epsilon]).to(self.device))

    def forward(self, input):
        output = SPDRectifiedFunction.apply(input, self.epsilon)
        return output



class SPD_UNET(nn.Module):
    def __init__(self, spd_size, time_size):
        super().__init__()
        self.time_dim = time_size
        #self.spd_size = spd_size

        self.transform = nn.ModuleList([
            SPDTransform(spd_size, spd_size, self.time_dim),                                # layer 1
            SPDTransform(spd_size, round(spd_size * (3/4)), self.time_dim),                 # layer 1.5

            SPDTransform(round(spd_size * (3/4)), round(spd_size * (3/4)), self.time_dim),  # layer 2
            SPDTransform(round(spd_size * (3/4)), round(spd_size * (2/3)), self.time_dim),  # layer 2.5

            SPDTransform(round(spd_size * (2/3)), round(spd_size * (1/2)), self.time_dim),  # layer 3
            SPDTransform(round(spd_size * (1/2)), round(spd_size * (1/2)), self.time_dim),  # layer 3.5

            SPDTransform(round(spd_size * (1/2)), round(spd_size * (1/4)), self.time_dim),  # layer 4
            SPDTransform(round(spd_size * (1/4)), round(spd_size * (1/4)), self.time_dim),  # layer 4.5

            SPDTransform(round(spd_size * (1/4)), round(spd_size * (1/6)), self.time_dim),  # layer 5
            SPDTransform(round(spd_size * (1/6)), round(spd_size * (1/6)), self.time_dim),  # layer 5.5

            SPDTransform(round(spd_size * (1/6)), round(spd_size * (1/4)), self.time_dim),  # layer 6
            SPDTransform(round(spd_size * (1/4)), round(spd_size * (1/4)), self.time_dim),  # layer 6.5

            SPDTransform(round(spd_size * (1/4)), round(spd_size * (1/2)), self.time_dim),  # layer 7
            SPDTransform(round(spd_size * (1/2)), round(spd_size * (1/2)), self.time_dim),  # layer 7.5

            SPDTransform(round(spd_size * (1/2)), round(spd_size * (2/3)), self.time_dim),  # layer 8
            SPDTransform(round(spd_size * (2/3)), round(spd_size * (2/3)), self.time_dim),  # layer 8.5

            SPDTransform(round(spd_size * (2/3)), round(spd_size * (3/4)), self.time_dim),  # layer 9
            SPDTransform(round(spd_size * (3/4)), round(spd_size * (3/4)), self.time_dim),  # layer 9.5

            SPDTransform(round(spd_size * (3/4)), spd_size, self.time_dim),                 # layer 10
            SPDTransform(spd_size, spd_size, self.time_dim),                                # layer 10.5

            SPDTransform(spd_size, spd_size, self.time_dim)])                               # layer 11
        

        self.rectification = nn.ModuleList([SPDRectified() for _ in range(len(self.transform)-1)])            #rectification list


    def unet_forward(self, x, t):

        for i in range(len(self.rectification)):
            x = self.transform[i](x, t)
            x = self.rectification[i](x)

        x = self.transform[-1](x, t)
        return x

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=t.device) / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1).double()

    def forward(self, x, t):
        t = t.unsqueeze(-1).float()
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forward(x, t)


