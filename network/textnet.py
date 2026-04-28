import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from cfglib.config import config as cfg
from network.layers.model_block import FPN
from network.layers.mamba_bt import CircularBiMambaBT
from util.misc import get_sample_point
from network.layers.gcn_utils import get_node_feature


class Evolution(nn.Module):
    def __init__(self, node_num, adj_num, is_training=True, device=None):
        super(Evolution, self).__init__()
        self.node_num = node_num
        self.adj_num = adj_num
        self.device = device
        self.is_training = is_training
        self.clip_dis = 16
        self.iter = 3
        self.adj = None

        for i in range(self.iter):
            evolve_model = CircularBiMambaBT(
                in_dim=36,
                hidden_dim=128,
                n_layers=3,
                num_heads=8,
                d_state=16,
                d_conv=4,
                expand=2,
            )
            self.__setattr__("evolve_gcn" + str(i), evolve_model)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def get_boundary_proposal(input=None, seg_preds=None, switch="gt"):
        if switch == "gt":
            inds = torch.where(input["ignore_tags"] > 0)
            init_polys = input["proposal_points"][inds]
        else:
            tr_masks = input["tr_mask"].cpu().numpy()
            tcl_masks = seg_preds[:, 0, :, :].detach().cpu().numpy() > cfg.threshold

            inds = []
            init_polys = []

            for bid, tcl_mask in enumerate(tcl_masks):
                ret, labels = cv2.connectedComponents(
                    tcl_mask.astype(np.uint8),
                    connectivity=8
                )

                for idx in range(1, ret):
                    text_mask = labels == idx
                    ist_id = int(np.sum(text_mask * tr_masks[bid]) / np.sum(text_mask)) - 1

                    inds.append([bid, ist_id])

                    poly = get_sample_point(
                        text_mask,
                        cfg.num_points,
                        cfg.approx_factor
                    )
                    init_polys.append(poly)

            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device)

        return init_polys, inds, None

    def get_boundary_proposal_eval(self, input=None, seg_preds=None):
        cls_preds = seg_preds[:, 0, :, :].detach().cpu().numpy()
        dis_preds = seg_preds[:, 1, :, :].detach().cpu().numpy()

        inds = []
        init_polys = []
        confidences = []

        for bid, dis_pred in enumerate(dis_preds):
            dis_mask = dis_pred > cfg.dis_threshold

            ret, labels = cv2.connectedComponents(
                dis_mask.astype(np.uint8),
                connectivity=8,
                ltype=cv2.CV_32S
            )

            for idx in range(1, ret):
                text_mask = labels == idx
                confidence = round(cls_preds[bid][text_mask].mean(), 3)

                min_area = 50 / (cfg.scale * cfg.scale)
                if np.sum(text_mask) < min_area or confidence < cfg.cls_threshold:
                    continue

                confidences.append(confidence)
                inds.append([bid, 0])

                poly = get_sample_point(
                    text_mask,
                    cfg.num_points,
                    cfg.approx_factor,
                    scales=np.array([cfg.scale, cfg.scale])
                )
                init_polys.append(poly)

        if len(inds) > 0:
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(
                input["img"].device,
                non_blocking=True
            )

            init_polys = torch.from_numpy(np.array(init_polys)).to(
                input["img"].device,
                non_blocking=True
            ).float()
        else:
            inds = torch.from_numpy(np.array(inds)).to(
                input["img"].device,
                non_blocking=True
            )

            init_polys = torch.from_numpy(np.array(init_polys)).to(
                input["img"].device,
                non_blocking=True
            ).float()

        return init_polys, inds, confidences

    def evolve_poly(self, evolve_model, cnn_feature, i_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)

        h = cnn_feature.size(2) * cfg.scale
        w = cnn_feature.size(3) * cfg.scale

        node_feats = get_node_feature(
            cnn_feature,
            i_it_poly,
            ind,
            h,
            w
        )

        offset = evolve_model(node_feats, self.adj).permute(0, 2, 1)
        offset = torch.clamp(offset, -self.clip_dis, self.clip_dis)

        i_poly = i_it_poly + offset

        if self.is_training:
            i_poly = torch.clamp(i_poly, 0, w - 1)
        else:
            i_poly[:, :, 0] = torch.clamp(i_poly[:, :, 0], 0, w - 1)
            i_poly[:, :, 1] = torch.clamp(i_poly[:, :, 1], 0, h - 1)

        return i_poly

    def forward(self, embed_feature, input=None, seg_preds=None, switch="gt"):
        if self.is_training:
            init_polys, inds, confidences = self.get_boundary_proposal(
                input=input,
                seg_preds=seg_preds,
                switch=switch
            )
        else:
            init_polys, inds, confidences = self.get_boundary_proposal_eval(
                input=input,
                seg_preds=seg_preds
            )

            if init_polys.shape[0] == 0:
                return [init_polys for _ in range(self.iter + 1)], inds, confidences

        py_preds = [init_polys]

        for i in range(self.iter):
            evolve_model = self.__getattr__("evolve_gcn" + str(i))
            init_polys = self.evolve_poly(
                evolve_model,
                embed_feature,
                init_polys,
                inds[0]
            )
            py_preds.append(init_polys)

        return py_preds, inds, confidences


class PFM_HFP(nn.Module):
    """
    Cross-scale pair fusion with high-frequency enhancement.

    Input:
        c5: up5, shape = (B, 256, Hf/16, Wf/16)
        c4: up4, shape = (B, 128, Hf/8,  Wf/8)
        c3: up3, shape = (B, 64,  Hf/4,  Wf/4)
        c2: up2, shape = (B, 32,  Hf/2,  Wf/2)

    Output:
        out: shape = (B, out_channels, Hf, Wf)
    """

    def __init__(
        self,
        c5=256,
        c4=128,
        c3=64,
        c2=32,
        mid_channels=64,
        out_channels=16,
        use_hfp=True
    ):
        super(PFM_HFP, self).__init__()

        self.use_hfp = use_hfp

        if use_hfp:
            from network.layers.hfp import HFP

            self.hfp5 = HFP(c5, ratio=(0.25, 0.25), isdct=True)
            self.hfp4 = HFP(c4, ratio=(0.25, 0.25), isdct=True)
            self.hfp3 = HFP(c3, ratio=(0.25, 0.25), isdct=True)
            self.hfp2 = HFP(c2, ratio=(0.25, 0.25), isdct=True)
        else:
            self.hfp5 = nn.Identity()
            self.hfp4 = nn.Identity()
            self.hfp3 = nn.Identity()
            self.hfp2 = nn.Identity()

        self.fuse_24 = nn.Sequential(
            nn.Conv2d(c2 + c4, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse_35 = nn.Sequential(
            nn.Conv2d(c3 + c5, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse_out = nn.Sequential(
            nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, c5, c4, c3, c2, target_hw):
        x5 = self.hfp5(c5)
        x4 = self.hfp4(c4)
        x3 = self.hfp3(c3)
        x2 = self.hfp2(c2)

        x4_to_x2 = F.interpolate(
            x4,
            size=x2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        x0 = self.fuse_24(torch.cat([x2, x4_to_x2], dim=1))

        x5_to_x3 = F.interpolate(
            x5,
            size=x3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        x1 = self.fuse_35(torch.cat([x3, x5_to_x3], dim=1))

        x1_to_x0 = F.interpolate(
            x1,
            size=x0.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        out = self.fuse_out(torch.cat([x0, x1_to_x0], dim=1))
        out = self.smooth(out)

        out = F.interpolate(
            out,
            size=target_hw,
            mode="bilinear",
            align_corners=False
        )

        return out


class TextNet(nn.Module):
    def __init__(self, backbone="vgg", is_training=True):
        super(TextNet, self).__init__()

        self.is_training = is_training
        self.backbone_name = backbone

        self.fpn = FPN(
            self.backbone_name,
            is_training=(not cfg.resume and is_training)
        )

        self.pfm_hfp = PFM_HFP(
            c5=256,
            c4=128,
            c3=64,
            c2=32,
            mid_channels=64,
            out_channels=16
        )

        self.ch_up1 = 32

        self.res_conv1 = nn.Conv2d(
            16,
            self.ch_up1,
            kernel_size=1,
            bias=False
        )
        self.res_bn = nn.BatchNorm2d(self.ch_up1)
        self.res_gamma = nn.Parameter(torch.ones(1, self.ch_up1, 1, 1))
        self.res_relu = nn.ReLU(inplace=True)

        self.seg_head = nn.Sequential(
            nn.Conv2d(self.ch_up1, 16, kernel_size=3, padding=2, dilation=2),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4),
            nn.PReLU(),
            nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
        )

        self.BPN = Evolution(
            cfg.num_points,
            adj_num=4,
            is_training=is_training,
            device=cfg.device
        )

    def load_model(self, model_path):
        print("Loading from {}".format(model_path))

        state_dict = torch.load(
            model_path,
            map_location=torch.device(cfg.device)
        )

        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        clean_state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }

        self.load_state_dict(
            clean_state_dict,
            strict=(not self.is_training)
        )

    def forward(self, input_dict, test_speed=False):
        output = {}

        b, c, h, w = input_dict["img"].shape

        if self.is_training or cfg.exp_name in ["ArT", "MLT2017", "MLT2019"] or test_speed:
            image = input_dict["img"]
        else:
            image = torch.zeros(
                (b, c, cfg.test_size[1], cfg.test_size[1]),
                dtype=torch.float32
            ).to(cfg.device)

            image[:, :, :h, :w] = input_dict["img"][:, :, :, :]

        up1, up2, up3, up4, up5 = self.fpn(image)

        hf = h // cfg.scale
        wf = w // cfg.scale

        up1 = up1[:, :, :hf, :wf]
        up2 = up2[:, :, :hf // 2, :wf // 2]
        up3 = up3[:, :, :hf // 4, :wf // 4]
        up4 = up4[:, :, :hf // 8, :wf // 8]
        up5 = up5[:, :, :hf // 16, :wf // 16]

        ff_up = self.pfm_hfp(
            up5,
            up4,
            up3,
            up2,
            target_hw=(hf, wf)
        )

        fb = up1
        tc = ff_up

        res = self.res_conv1(tc)
        res = self.res_bn(res)
        res = self.res_gamma * res

        f_tilde = self.res_relu(fb + res)

        preds = self.seg_head(f_tilde)

        fy_preds = torch.cat(
            [
                torch.sigmoid(preds[:, 0:2, :, :]),
                preds[:, 2:4, :, :]
            ],
            dim=1
        )

        cnn_feats = torch.cat([f_tilde, fy_preds], dim=1)

        py_preds, inds, confidences = self.BPN(
            cnn_feats,
            input=input_dict,
            seg_preds=fy_preds,
            switch="gt"
        )

        output["fy_preds"] = fy_preds
        output["py_preds"] = py_preds
        output["inds"] = inds
        output["confidences"] = confidences

        return output