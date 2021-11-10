import itertools
from functools import partial

from torch import multiprocessing

import torch
from torch import nn as nn
from torch.nn import functional as F

from .utils import l2norm


def dot_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    im = l2norm(im)
    s = l2norm(s)
    return im.mm(s.t())

def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class Contrastive(nn.Module):
    def __init__(self, margin=0, measure=False, max_violation=False):
        super(Contrastive, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'dot':
            self.sim = dot_sim

        self.max_violation = max_violation

    def compute_contrastive_loss(self, scores):
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class AlignmentContrastiveLoss(Contrastive):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, aggregation='sum-max-sentences', return_similarity_mat=False, force_alignment=False):
        super(AlignmentContrastiveLoss, self).__init__(margin, measure, max_violation)
        self.aggregation = aggregation
        self.force_alignment = force_alignment
        self.return_similarity_mat = return_similarity_mat
        if aggregation == 'emd':
            self.emd = EMDLoss()

    def compute_strong_alignment_loss(self, alignments, mask):
        assert self.aggregation=='sum-max-images' or self.aggregation=='sum-max-images-sentences'
        # take only the diagonal items, the ones that should match
        assert alignments.shape[0] == alignments.shape[1]
        diag = torch.eye(alignments.shape[0]).bool()
        matching_alignments = alignments[diag]  # bs x im x sent
        matching_alignments_images = matching_alignments.max(1)[0] # bs x sent
        mask = mask[diag]
        mask = ~mask
        mask_images = mask.any(1)
        matching_alignments_images = matching_alignments_images[mask_images]
        loss = (1 - matching_alignments_images).mean()
        if self.aggregation=='sum-max-images-sentences':
            matching_alignments_sent = matching_alignments.max(2)[0]  # bs x imgs
            mask_sentences = mask.any(2)
            matching_alignments_sent = matching_alignments_sent[mask_sentences]
            loss += (1 - matching_alignments_sent).mean()
        return loss

    # def compute_strong_alignment_loss(self, alignments, mask):
    #     assert self.aggregation=='sum-max-images'
    #     # take only the diagonal items, the ones that should match
    #     assert alignments.shape[0] == alignments.shape[1]
    #     diag = torch.eye(alignments.shape[0]).bool()
    #     matching_alignments = alignments[diag]  # bs x im x sent
    #     sorted_regions, _ = torch.sort(matching_alignments, dim=1, descending=True) # bs x im x sent
    #     # pos_neg_regions = sorted_regions[:, :2, :]
    #
    #     # hard negative sampling
    #     loss = (1 - sorted_regions[:, 0, :])**2 + (-1 - sorted_regions[:, -1, :])**2   # bs x sent  # (self.margin - pos_neg_regions[:, 0, :] + pos_neg_regions[:, 1, :]).clamp(min=0)
    #     mask = mask[diag]
    #     mask = ~mask
    #     mask = mask.any(1)
    #     loss = loss[mask]
    #     loss = loss.mean()
    #
    #     return loss

    def forward(self, im_set, s_seq, im_len, s_len):
        # im_set = im_set.permute(1, 0, 2)    # B x S_im x dim
        # s_seq = s_seq.permute(1, 0, 2)     # B x S_s x dim

        # do not consider cls and eos tokens
        im_set = im_set[:, 1:, :]
        s_seq = s_seq[:, 1:-2, :]
        im_len = [l - 1 for l in im_len]
        s_len = [l - 3 for l in s_len]

        im_set_batch = im_set.size(0)
        im_set_len = im_set.size(1)
        s_seq_batch = s_seq.size(0)
        s_seq_len = s_seq.size(1)

        im_set = im_set.unsqueeze(1).expand(-1, s_seq_batch, -1, -1)  # B x B x S_im x dim
        s_seq = s_seq.unsqueeze(0).expand(im_set_batch, -1, -1, -1) # B x B x S_s x dim
        alignments = torch.matmul(im_set, s_seq.permute(0, 1, 3, 2))  # B x B x S_im x S_s
        # alignments = F.relu(alignments)

        # compute mask for the alignments tensor
        im_len_mask = torch.zeros(im_set_batch, im_set_len).bool()
        im_len_mask = im_len_mask.to(im_set.device)
        for im, l in zip(im_len_mask, im_len):
            im[l:] = True
        im_len_mask = im_len_mask.unsqueeze(2).unsqueeze(1).expand(-1, s_seq_batch, -1, s_seq_len)

        s_len_mask = torch.zeros(s_seq_batch, s_seq_len).bool()
        s_len_mask = s_len_mask.to(im_set.device)
        for sm, l in zip(s_len_mask, s_len):
            sm[l:] = True
        s_len_mask = s_len_mask.unsqueeze(1).unsqueeze(0).expand(im_set_batch, -1, im_set_len, -1)

        alignment_mask = im_len_mask | s_len_mask
        alignments.masked_fill_(alignment_mask, value=0)
        # alignments = F.relu(alignments)
        # alignments = F.normalize(alignments,p=2, dim=2)

        if self.aggregation == 'emd':
            aggr_similarity = self.emd(alignments, im_set, s_seq, im_len, s_len)
        if self.aggregation == 'sum':
            aggr_similarity = alignments.sum(dim=(2,3))
        elif self.aggregation == 'mean':
            aggr_similarity = alignments.mean(dim=(2,3))
        elif self.aggregation == 'sum-max-images':
            aggr_similarity = alignments.max(2)[0].sum(2)
        elif self.aggregation == 'sum-sum-images':
            aggr_similarity = alignments.sum((2,3))
        elif self.aggregation == 'avg-max-images':
            aggr_similarity = alignments.max(2)[0].sum(2)
            expanded_len = torch.FloatTensor(s_len).to(alignments.device).unsqueeze(0).expand(len(im_len), -1)
            aggr_similarity /= expanded_len
        elif self.aggregation == 'sum-max-images-sentences':
            im = alignments.max(2)[0].sum(2)
            s = alignments.max(3)[0].sum(2)
            aggr_similarity = im + s
        elif self.aggregation == 'sum-max-sentences':
            aggr_similarity = alignments.max(3)[0].sum(2)
        elif self.aggregation == 'scan-sentences':
            norm_alignments = F.relu(alignments)
            norm_alignments = F.normalize(norm_alignments,p=2, dim=2)
            weights = norm_alignments.masked_fill(alignment_mask, value=float('-inf'))
            weights = torch.softmax(weights, dim=3)

            weights = weights.unsqueeze(3)  # B x B x im x 1 x s
            s_seq_ext = s_seq.unsqueeze(2).expand(-1, -1, im_set_len, -1, -1)
            att_vector = torch.matmul(weights, s_seq_ext)  # B x B x im x 1 x dim
            att_vector = att_vector.squeeze(3)
            new_alignments = F.cosine_similarity(im_set, att_vector, dim=3)  # B x B x im
            new_alignments.masked_fill_(im_len_mask[:, :, :, 0], value=0)

            aggr_similarity = new_alignments.sum(2)

        if self.return_similarity_mat:
            return aggr_similarity
        else:
            loss = self.compute_contrastive_loss(aggr_similarity)
            force_alignment_loss = self.compute_strong_alignment_loss(alignments, alignment_mask) if self.force_alignment else None
            return loss, force_alignment_loss


class ContrastiveLoss(Contrastive):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'dot':
            self.sim = dot_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        return self.compute_contrastive_loss(scores)


class PermInvMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # @staticmethod
    # def batched_cosine_sim(im, s):
    #     """Cosine similarity between all the image and sentence pairs
    #     """
    #     im = F.normalize(im, p=2, dim=2)
    #     s = F.normalize(s, p=2, dim=2)
    #     return im.mm(s.permute(0, 2, 1))

    def forward(self, im, s):
        dist_matrix = torch.cdist(im, s, p=2)
        row_sum = F.softmin(dist_matrix, dim=2).max(dim=2)[0].sum(dim=1)
        col_sum = F.softmin(dist_matrix, dim=1).max(dim=1)[0].sum(dim=1)
        loss = 2*torch.Tensor([dist_matrix.shape[1]]).to(im.device) - row_sum - col_sum
        loss = loss.mean()
        return loss


class EMDLoss(nn.Module):
    pool = None

    def __init__(self, solver='opencv', metric='cosine', temperature=None, ncpus=16):
        super().__init__()
        self.solver = solver
        self.metric = metric
        self.temperature = temperature
        if EMDLoss.pool is None:
            EMDLoss.pool = multiprocessing.Pool(ncpus)

    # def forward(self, im_set, s_seq, im_len, s_len):
    #     return self.emd_forward_1shot(im_set, s_seq, im_len, s_len)

    # def get_weight_vector(self, A, B):
    #
    #     M = A.shape[0]
    #     N = B.shape[0]
    #
    #     B = F.adaptive_avg_pool2d(B, [1, 1])
    #     B = B.repeat(1, 1, A.shape[2], A.shape[3])
    #
    #     A = A.unsqueeze(1)
    #     B = B.unsqueeze(0)
    #
    #     A = A.repeat(1, N, 1, 1, 1)
    #     B = B.repeat(M, 1, 1, 1, 1)
    #
    #     combination = (A * B).sum(2)
    #     combination = combination.view(M, N, -1)
    #     combination = F.relu(combination) + 1e-3
    #     return combination

    def get_weight_vector(self, A, B, A_len):
        bs_A = len(A)
        bs_B = len(B)
        weight = torch.zeros(bs_A, A_len)
        for m, c_len in zip(weight, A):
            m[:c_len] = 1.0
        weight = weight.unsqueeze(1).expand(-1, bs_B, -1)
        return weight

    def forward(self, similarity_map, im_set, s_seq, im_len, s_len):
        # im_set = im_set.squeeze(0)

        # weight_1 = self.get_weight_vector(s_seq, im_set)
        # weight_2 = self.get_weight_vector(im_set, s_seq)

        weight_1 = self.get_weight_vector(im_len, s_len, similarity_map.shape[2]).contiguous()
        weight_2 = self.get_weight_vector(s_len, im_len, similarity_map.shape[3]).contiguous()

        # proto = self.normalize_feature(proto)
        # query = self.normalize_feature(query)

        # similarity_map = self.get_similiarity_map(im_set, s_seq)
        if self.solver == 'opencv' or (not self.training):
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv')
        else:
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='qpth')
        return logits

    def get_emd_distance(self, similarity_map, weight_1, weight_2, solver='opencv'):
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        num_node=weight_1.shape[-1]

        if solver == 'opencv':  # use openCV solver
            # for i in range(num_query):
            #     for j in range(num_proto):
            #         _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])
            #
            #         similarity_map[i, j, :, :] =(similarity_map[i, j, :, :])*torch.from_numpy(flow).cuda()

            # Parallelized version:
            similarity_map_sh = similarity_map.detach().cpu()  # .share_memory_()
            weight_1_sh = weight_1.detach().cpu()
            weight_2_sh = weight_2.detach().cpu()
            flows = EMDLoss.pool.starmap(partial(emd_single_pass, similarity_map=similarity_map_sh, weight_1=weight_1_sh, weight_2=weight_2_sh),
                                         itertools.product(range(num_query), range(num_proto)))
            flows = [torch.from_numpy(x) for x in flows]
            flows = torch.stack(flows).view(num_query, num_proto, similarity_map.shape[2], similarity_map.shape[3]).cuda()
            similarity_map *= flows
            logitis = similarity_map.sum(-1).sum(-1)
            if self.temperature is not None:
                temperature=(self.temperature/num_node)
                logitis *= temperature
            return logitis

        elif solver == 'qpth':
            weight_2 = weight_2.permute(1, 0, 2)
            similarity_map = similarity_map.view(num_query * num_proto, similarity_map.shape[-2],
                                                 similarity_map.shape[-1])
            weight_1 = weight_1.view(num_query * num_proto, weight_1.shape[-1])
            weight_2 = weight_2.reshape(num_query * num_proto, weight_2.shape[-1])

            _, flows = emd_inference_qpth(1 - similarity_map, weight_1, weight_2,form='QP') #, l2_strength=self.args.l2_strength)

            logitis=(flows*similarity_map).view(num_query, num_proto,flows.shape[-2],flows.shape[-1])
            logitis = logitis.sum(-1).sum(-1)
            if self.temperature is not None:
                temperature = (self.temperature / num_node)
                logitis *= temperature
        else:
            raise ValueError('Unknown Solver')

        return logitis

    def normalize_feature(self, x):
        if self.args.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
            return x
        else:
            return x

    def get_similiarity_map(self, proto, query):
        way = proto.shape[0]
        num_query = query.shape[0]
        query = query.view(query.shape[0], query.shape[1], -1)
        proto = proto.view(proto.shape[0], proto.shape[1], -1)

        proto = proto.unsqueeze(0).expand(num_query, -1, -1, -1)
        query = query.unsqueeze(1).expand(-1, way, -1, -1)
        proto = proto.permute(0, 1, 3, 2)
        query = query.permute(0, 1, 3, 2)
        feature_size = proto.shape[-2]

        if self.metric == 'cosine':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.expand(-1, -1, -1, feature_size, -1)
            similarity_map = F.cosine_similarity(proto, query, dim=-1)
        if self.metric == 'l2':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.expand(-1, -1, -1, feature_size, -1)
            similarity_map = (proto - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map

        return similarity_map


def emd_one_row(i, similarity_map, weight_1, weight_2):
    num_proto = similarity_map.shape[1]
    flows = []
    for j in range(num_proto):
        _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])
        # similarity_map[i, j, :, :] = (similarity_map[i, j, :, :]) * torch.from_numpy(flow).cuda()
        flows.append(torch.from_numpy(flow))
    flow = torch.stack(flows)
    return flow

def emd_single_pass(i, j, similarity_map, weight_1, weight_2):
    _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])
    # similarity_map[i, j, :, :] = (similarity_map[i, j, :, :]) * torch.from_numpy(flow).cuda()
    return flow
