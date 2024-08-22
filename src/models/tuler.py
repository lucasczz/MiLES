import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class TULER(nn.Module):

    def __init__(
        self,
        vocab_size,
        num_users,
        dim_inp,
        dim_out,
        num_layers=1,
        dropout=0.5,
        model_type="LSTM",
        bidirectional=False,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, dim_inp)
        if model_type == "LSTM":
            self.model = nn.LSTM(
                input_size=dim_inp,
                hidden_size=dim_out,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif model_type == "RNN":
            self.model = nn.RNN(
                input_size=dim_inp,
                hidden_size=dim_out,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif model_type == "GRU":
            self.model = nn.GRU(
                input_size=dim_inp,
                hidden_size=dim_out,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        self.model_type = model_type
        # in case of bidrectional
        self.fc1 = nn.Linear(dim_out * 2, dim_out)
        self.fc2 = nn.Linear(dim_out, num_users)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.bi = bidirectional

    def forward(self, input_tensor: torch.Tensor, org_length: torch.Tensor):
        # input_tensor has a shape like (num of smaples per batch, seq)
        model_input = self.token_emb(
            input_tensor
        )  # This results in a tensor of shape (num of smaples per batch, seq, dim_inp)
        model_input = pack_padded_sequence(
            model_input, torch.flatten(org_length).to("cpu"), True, False
        )
        if self.model_type == "LSTM":
            output, (hidden, cell) = self.model(model_input)
        else:
            output, hidden = self.model(model_input)
        if self.bi:
            hidden = self.fc1(torch.cat((hidden[-1, :, :], hidden[-2, :, :]), dim=-1))
            output = self.fc2(hidden)
        else:
            output, output_lengths = pad_packed_sequence(output, batch_first=True)
            last_seq_idxs = torch.LongTensor([x - 1 for x in output_lengths])
            last_seq_items = output[range(output.shape[0]), last_seq_idxs, :]
            # print(last_seq_items.size())
            output = self.fc2(last_seq_items)
        return self.dropout(output)
