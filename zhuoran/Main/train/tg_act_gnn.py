import copy
import logging

import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR

from model_utils import MLP_3, CG_stacked

# Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim


logger = logging.getLogger(__name__)
copy_func = copy.deepcopy
version = 0


# Define the GCN model
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(8, 16)  # GCN layer with input dimension 8 and output dimension 16
        self.linear = nn.Linear(16 * 5, 89 * 5)  # Linear layer for the output

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        hidden = self.conv1(x, edge_index)  # Get hidden state after GCN but before activation
        x = F.relu(hidden)  # Applying ReLU activation function
        x = x.view(-1, 16*5)  # Flattening the output for the linear layer
        x = self.linear(x)
        return x.view(-1, 89, 5), hidden  # Return output and hidden state



# Load the saved model for validation
checkpoint = torch.load('gcn_model_checkpoint.pth')
model = GCN()  # Re-instantiate the model
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation model
optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])




def act_loss(pred, gt):
    MSE = torch.nn.MSELoss(reduction='none')
    L1 = torch.nn.L1Loss(reduction='none')
    CLS = torch.nn.CrossEntropyLoss()

    prob_pred = pred['prob']
    velo_pred = pred['velo']
    pos_pred = pred['pos']
    heading_pred = pred['heading']

    pos_gt = gt['gt_pos'].unsqueeze(1).repeat(1, 6, 1, 1)
    velo_gt = gt['gt_vel'].unsqueeze(1).repeat(1, 6, 1, 1)
    heading_gt = gt['gt_heading'].unsqueeze(1).repeat(1, 6, 1)

    pred_end = pos_pred[:, :, -1]
    gt_end = pos_gt[:, :, -1]
    dist = MSE(pred_end, gt_end).mean(-1)
    min_index = torch.argmin(dist, dim=-1)

    cls_loss = CLS(prob_pred, min_index)

    pos_loss = MSE(pos_gt, pos_pred).mean(-1).mean(-1)
    fde = MSE(pos_gt, pos_pred).mean(-1)[..., -1]
    pos_loss = torch.gather(pos_loss, dim=1, index=min_index.unsqueeze(-1)).mean()
    fde = torch.gather(fde, dim=1, index=min_index.unsqueeze(-1)).mean()

    velo_loss = MSE(velo_gt, velo_pred).mean(-1).mean(-1)
    velo_loss = torch.gather(velo_loss, dim=1, index=min_index.unsqueeze(-1)).mean()

    heading_loss = L1(heading_gt, heading_pred).mean(-1)
    heading_loss = torch.gather(heading_loss, dim=1, index=min_index.unsqueeze(-1)).mean()

    loss_sum = pos_loss + velo_loss + heading_loss + cls_loss

    loss_dict = {
        'cls_loss': cls_loss,
        'velo_loss': velo_loss,
        'heading_loss': heading_loss,
        'fde': fde,
        'pos_loss': pos_loss
    }
    return loss_sum, loss_dict


class actuator(pl.LightningModule):
    """ A transformer model with wider latent space """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        hidden_dim = 1024
        self.CG_agent = CG_stacked(5, hidden_dim)
        self.CG_line = CG_stacked(5, hidden_dim)
        self.CG_all = CG_stacked(5, hidden_dim * 2)
        self.agent_encode = MLP_3([8, 256, 512, hidden_dim])
        self.line_encode = MLP_3([4, 256, 512, hidden_dim])
        self.type_embedding = nn.Embedding(20, hidden_dim)
        self.traf_embedding = nn.Embedding(4, hidden_dim)
        self.anchor_embedding = nn.Embedding(6, hidden_dim * 2)

        self.pred_len = 89

        self.velo_head = MLP_3([hidden_dim * 2, hidden_dim, 256, self.pred_len * 2])
        self.pos_head = MLP_3([hidden_dim * 2, hidden_dim, 256, self.pred_len * 2])
        self.angle_head = MLP_3([hidden_dim * 2, hidden_dim, 256, self.pred_len])
        self.prob_head = MLP_3([hidden_dim * 2, hidden_dim, 256, 1])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss, loss_dict = act_loss(pred, batch)
        loss_dict = {'train/' + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss, loss_dict = act_loss(pred, batch)
        loss_dict = {'val/' + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = MultiStepLR(optimizer, milestones=[50, 100], gamma=0.2, verbose=True)
        return [optimizer], [scheduler]

    def forward(self, data):
        agent = data['agent']
        agent_mask = data['agent_mask']

        all_vec = torch.cat([data['center'], data['cross'], data['bound']], dim=-2)
        line_mask = torch.cat([data['center_mask'], data['cross_mask'], data['bound_mask']], dim=1)

        polyline = all_vec[..., :4]
        polyline_type = all_vec[..., 4].to(torch.int)
        polyline_traf = all_vec[..., 5].to(torch.int)
        polyline_type_embed = self.type_embedding(polyline_type)
        polyline_traf_embed = self.traf_embedding(polyline_traf)

        agent_enc = self.agent_encode(agent)
        line_enc = self.line_encode(polyline) + polyline_traf_embed + polyline_type_embed
        b, a, d = agent_enc.shape

        device = agent_enc.device

        context_agent = torch.ones([b, d]).to(device)
        agent_enc, context_agent = self.CG_agent(agent_enc, context_agent, agent_mask)

        line_enc, context_line = self.CG_line(line_enc, context_agent, line_mask)

        all_context = torch.cat([context_agent, context_line], dim=-1)

        anchors = self.anchor_embedding.weight.unsqueeze(0).repeat(b, 1, 1)
        mask = torch.ones(*anchors.shape[:-1]).to(device)
        pred_embed, _ = self.CG_all(anchors, all_context, mask)



        prob_pred = self.prob_head(pred_embed).squeeze(-1)
        velo_pred = self.velo_head(pred_embed).view(b, 6, self.pred_len, 2)
        pos_pred = self.pos_head(pred_embed).view(b, 6, self.pred_len, 2).cumsum(-2)
        heading_pred = self.angle_head(pred_embed).view(b, 6, self.pred_len).cumsum(-1)

        pred = {
            'prob': prob_pred,
            'velo': velo_pred,
            'pos': pos_pred,
            'heading': heading_pred
        }

        return pred
