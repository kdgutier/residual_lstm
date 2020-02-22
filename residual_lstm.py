import torch
import torch.nn as nn
import torch.jit as jit


class ResLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(ResLSTMCell, self).__init__()
        self.register_buffer('input_size', torch.Tensor([input_size]))
        self.register_buffer('hidden_size', torch.Tensor([hidden_size]))
        self.weight_ii = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_ic = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ii = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ic = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(1 * hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(1 * hidden_size))
        self.weight_ir = nn.Parameter(torch.randn(hidden_size, input_size))
        #self.dropout_layer = nn.Dropout(dropout)
        self.dropout = dropout

    @jit.script_method
    def forward(self, input, hidden):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = hidden[0].squeeze(0), hidden[1].squeeze(0)

        ifo_gates = (torch.mm(input, self.weight_ii.t()) + self.bias_ii +
                     torch.mm(hx, self.weight_ih.t()) + self.bias_ih +
                     torch.mm(cx, self.weight_ic.t()) + self.bias_ic)
        ingate, forgetgate, outgate = ifo_gates.chunk(3, 1)
        
        cellgate = torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        ry = torch.tanh(cy)

        if self.input_size == self.hidden_size:
          hy = outgate * (ry + input)
        else:
          hy = outgate * (ry + torch.mm(input, self.weight_ir.t()))
        return hy, (hy, cy)

class ResLSTMLayer(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(ResLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.cell = LSTMCell(input_size, hidden_size, dropout=0.)
        self.cell = ResLSTMCell(input_size, hidden_size, dropout=0.)

    @jit.script_method
    def forward(self, input, hidden):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, hidden = self.cell(inputs[i], hidden)
            outputs += [out]
        outputs = torch.stack(outputs)
        print("outputs.size()", outputs.size())
        return outputs, hidden