import math
import torch
import Constants
import numpy as np
import torch.nn as nn


class DucDiff(nn.Module):
    def __init__(self, args):
        super(DucDiff, self).__init__()
        self.user_size = args.user_size
        self.hidden_size = args.hidden_size
        self.att_head = args.att_head
        self.user_embeddings = nn.Embedding(args.user_size, args.hidden_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(args.emb_dropout)
        self.position_emb = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        
        self.ib_past = VIB(in_feat=args.hidden_size, out_feat= args.compress_emb, num_class=args.user_size)
        self.ib_future = VIB(in_feat=args.hidden_size, out_feat= args.compress_emb, num_class=args.user_size)

        self.ib = VIB(in_feat=args.hidden_size, out_feat= args.compress_emb, num_class=args.user_size)

        self.classifier_noib_past = nn.Linear(args.hidden_size, args.user_size)
        self.classifier_noib_future = nn.Linear(args.hidden_size, args.user_size)

        self.linear_emb_past = nn.Linear(args.hidden_size, args.compress_emb, bias=False)
        self.linear_emb_future = nn.Linear(args.hidden_size, args.compress_emb, bias=False)

        self.optimizerAdam = torch.optim.Adam(self.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09)
        self.optimizer = ScheduledOptim(self.optimizerAdam, self.hidden_size, args.n_warmup_steps)

        self.loss_ce = nn.CrossEntropyLoss(size_average=False, ignore_index=0)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, cas_seq, label):

        past = cas_seq
        future = label

        mask_past = (past == Constants.PAD)
        mask_future = (future == Constants.PAD)

        position_ids = torch.arange(cas_seq.size(1), dtype=torch.long, device=cas_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(cas_seq)
        position_emb = self.position_emb(position_ids)

        past_emb = self.user_embeddings(past)
        future_emb = self.user_embeddings(future)
        diff_past_emb = self.LayerNorm(self.emb_dropout(past_emb + position_emb))
        diff_future_emb = self.LayerNorm(self.emb_dropout(future_emb + position_emb))


        past_hidden, past_dis = self.past_GD(diff_past_emb, future_emb, mask_past)
        future_hidden, future_dis = self.future_GD(diff_future_emb, past_emb, mask_future)
        
        past_output_ib, past_ib_feature = self.ib(past_hidden)
        future_output_ib, future_ib_feature = self.ib(future_hidden) # [batch,seq,user_size]


        past_output_noib = torch.matmul(past_hidden, self.user_embeddings.weight.t())
        future_output_noib = torch.matmul(future_hidden, self.user_embeddings.weight.t())

        past_output_ib = torch.matmul(past_ib_feature, self.linear_emb_past(self.user_embeddings.weight).t())
        future_output_ib = torch.matmul(future_ib_feature, self.linear_emb_future(self.user_embeddings.weight).t())

        mask = get_previous_user_mask(cas_seq.cuda(), self.user_size)
        past_output_noib_mask = (past_output_noib + mask).view(-1, past_output_noib.size(-1))


        return past_output_noib_mask, past_output_noib.view(-1, future_output_ib.size(-1)), future_output_noib.view(-1, future_output_ib.size(-1)), \
            past_output_ib.view(-1, future_output_ib.size(-1)), future_output_ib.view(-1, future_output_ib.size(-1))

    def model_prediction(self, cas_seq):
        mask = (cas_seq == Constants.PAD)

        position_ids = torch.arange(cas_seq.size(1), dtype=torch.long, device=cas_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(cas_seq)
        position_emb = self.position_emb(position_ids)
        cas_emb = self.user_embeddings(cas_seq)
        cas_emb = self.LayerNorm(cas_emb + position_emb)

        noise_x_t = torch.randn_like(cas_emb[:, :, :])
        hidden = self.past_GD.reverse_p_sample(cas_emb, noise_x_t, mask)
        hidden = torch.matmul(hidden, self.user_embeddings.weight.t())
        mask = get_previous_user_mask(cas_seq.cuda(), self.user_size)
        scores = (hidden + mask).view(-1, hidden.size(-1))
        return scores

