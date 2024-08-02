import math
import torch
import Constants
import numpy as np
import torch.nn as nn

from utils import ScheduledOptim, get_previous_user_mask
from step_sampler import create_named_schedule_sampler
from TransformerBlock import TransformerBlock, LayerNorm, SiLU
from beta_schedule import linear_beta_schedule, cosine_beta_schedule, exp_beta_schedule, betas_for_alpha_bar


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)

class Chanel_Compress(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Chanel_Compress, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_feat, math.floor(in_feat * 0.9)),
            nn.LayerNorm(math.floor(in_feat * 0.9)),
            nn.ReLU(),
            nn.Linear(math.floor(in_feat * 0.9), math.floor(in_feat * 0.85)),
            nn.LayerNorm(math.floor(in_feat * 0.85)),
            nn.ReLU(),
            nn.Linear(math.floor(in_feat * 0.85), out_feat)
        )

    def forward(self, x):
        x = x.reshape(-1, x.size(2))
        x = self.model(x)
        return x

class VIB(nn.Module):
    def __init__(self, in_feat, out_feat, num_class):
        super(VIB, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bottleneck = Chanel_Compress(in_feat=self.in_feat, out_feat=self.out_feat)

        self.classifier = nn.Sequential(
            nn.Linear(self.out_feat, math.floor(self.out_feat * 0.9)),
            nn.LayerNorm(math.floor(self.out_feat * 0.9)),
            nn.ReLU(),
            nn.Linear(math.floor(self.out_feat * 0.9), math.floor(self.out_feat * 0.85)),
            nn.LayerNorm(math.floor(self.out_feat * 0.85)),
            nn.ReLU(),
            nn.Linear(math.floor(self.out_feat * 0.85), num_class)
        )
        self.classifier.apply(weights_init_classifier)

    def forward(self, v):
        batch, seq, dim = v.size()
        z_given_v = self.bottleneck(v)
        p_y_given_z = self.classifier(z_given_v)
        return p_y_given_z.reshape(batch, seq, -1), z_given_v.reshape(batch, seq, -1)

class TransEncoder(nn.Module):
    def __init__(self, args, reverse):
        super(TransEncoder, self).__init__()
        self.hidden_size = args.hidden_size
        self.att_head = args.att_head
        self.reverse = reverse
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 2), SiLU(),
                                        nn.Linear(self.hidden_size * 2, self.hidden_size))
        self.att1 = TransformerBlock(input_size=self.hidden_size, n_heads=self.att_head, attn_dropout=args.dropout, reverse=self.reverse)
        self.att2 = TransformerBlock(input_size=self.hidden_size, n_heads=self.att_head, attn_dropout=args.dropout, reverse=self.reverse)
        self.att3 = TransformerBlock(input_size=self.hidden_size, n_heads=self.att_head, attn_dropout=args.dropout, reverse=self.reverse)
        
        self.lam = 0.001
        self.dropout = nn.Dropout(args.emb_dropout)
        self.norm = LayerNorm(self.hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
        opt = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(opt), torch.sin(opt)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, cas_emb, x_t, t, mask):

        time_emb = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        time_emb = time_emb.unsqueeze(1).repeat(1, cas_emb.size(1), 1)
        x_t = x_t + time_emb
        lam = torch.normal(mean=torch.full(cas_emb.shape, self.lam), std=torch.full(cas_emb.shape, self.lam)).to(
            x_t.device)

        z_emb = cas_emb + lam * x_t
        hidden, kl_dis = self.att1(z_emb, z_emb, z_emb, mask)

        hidden, kl_dis = self.att2(hidden, hidden, hidden, mask)
        return hidden, kl_dis


class GaussianDiffusion(nn.Module):
    def __init__(self, args, reverse):
        super(GaussianDiffusion, self).__init__()
        self.steps = args.steps
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end
        self.schedule_sampler_name = args.schedule_sampler_name
        self.beta_sche = args.beta_sche
        self.betas = self.get_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)  # alpha_bar
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.Encoder = TransEncoder(args, reverse)
        self.t_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.steps)
        self.rescale_timesteps = args.rescale_timesteps

    def get_betas(self):
        if self.beta_sche == 'linear':
            betas = linear_beta_schedule(timesteps=self.steps, beta_start=self.beta_start, beta_end=self.beta_end)
        elif self.beta_sche == 'exp':
            betas = exp_beta_schedule(timesteps=self.steps)
        elif self.beta_sche =='cosine':
            betas = cosine_beta_schedule(timesteps=self.steps)
        elif self.beta_sche =='sqrt':
            betas = torch.tensor(betas_for_alpha_bar(self.steps, lambda t: 1 - np.sqrt(t + 0.0001),)).float()
        else:
            raise NotImplementedError()
        betas = np.array(betas, dtype=np.float64)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        return betas

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_sample(self, x_start, t, noise=None, mask=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        assert noise.shape == x_start.shape
        x_t = (
                self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

        if mask is None:
            return x_t
        else:
            mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return torch.where(mask == 0, x_start, x_t)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  # mu_t
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, cas_emb, x_t, t, mask):
        hidden, kl_dis = self.Encoder(cas_emb, x_t, self._scale_timesteps(t), mask)
        x_0 = hidden
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x_t.shape)
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)
        return model_mean, model_log_variance

    def p_sample(self, cas_emb, noise_x_t, t, mask):
        model_mean, model_log_variance = self.p_mean_variance(cas_emb, noise_x_t, t, mask)
        noise = torch.randn_like(noise_x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1))))
        sample_xt = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return sample_xt


    def reverse_p_sample(self, cas_emb, noise_x_t, mask):
        device = next(self.Encoder.parameters()).device
        indices = list(range(self.steps))[::-1]
        for i in indices:
            t = torch.tensor([i] * cas_emb.shape[0], device=device)
            with torch.no_grad():
                noise_x_t = self.p_sample(cas_emb, noise_x_t, t, mask)
        return noise_x_t

    def forward(self, cas_emb, label_emb, mask):
        t, weights = self.t_sampler.sample(cas_emb.shape[0], cas_emb.device)
        noise = torch.randn_like(label_emb)
        x_t = self.q_sample(label_emb, t, noise=noise, mask=mask)
        hidden, kl_dis = self.Encoder(cas_emb, x_t, t, mask)
        return hidden, kl_dis


class DiffCas(nn.Module):
    def __init__(self, args):
        super(DiffCas, self).__init__()
        self.user_size = args.user_size
        self.hidden_size = args.hidden_size
        self.att_head = args.att_head
        self.user_embeddings = nn.Embedding(args.user_size, args.hidden_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(args.emb_dropout)
        self.position_emb = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.past_GD = GaussianDiffusion(args, reverse=False)
        self.future_GD = GaussianDiffusion(args, reverse=True)
        
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

