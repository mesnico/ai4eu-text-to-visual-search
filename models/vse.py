import torch
import torch.nn.init
import torch.backends.cudnn as cudnn

from models.loss import ContrastiveLoss
from models.text import EncoderText
from models.visual import EncoderImage


class VSE(torch.nn.Module):
    """
    rkiros/uvs model
    """

    def __init__(self, config):
        # tutorials/09 - Image Captioning
        # Build Models
        super().__init__()
        self.img_enc = EncoderImage(config)
        self.txt_enc = EncoderText(config)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=config['training']['margin'],
                                         measure=config['training']['measure'],
                                         max_violation=config['training']['max-violation'])

        self.Eiters = 0

    # def state_dict(self):
    #     state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
    #     return state_dict
    #
    # def load_state_dict(self, state_dict):
    #     self.img_enc.load_state_dict(state_dict[0])
    #     self.txt_enc.load_state_dict(state_dict[1])
    #
    # def train_start(self):
    #     """switch to train mode
    #     """
    #     self.img_enc.train()
    #     self.txt_enc.train()
    #
    # def val_start(self):
    #     """switch to evaluate mode
    #     """
    #     self.img_enc.eval()
    #     self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, None, None

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        return loss

    def forward(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        # assert self.training()
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)

        # compute the embeddings
        img_emb, cap_emb, _, _ = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        loss = self.forward_loss(img_emb, cap_emb)
        return {'tot_loss': loss}
