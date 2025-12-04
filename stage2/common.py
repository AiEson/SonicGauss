import os
import re
import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData


def load_ply(path, opacity_threshold: float = 0.02):
    """Load PLY file and convert SH from degree 3 to degree 1."""
    plydata = PlyData.read(path)
    
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    
    if len(extra_f_names) == 3 * (16 - 1):  # Degree 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((xyz.shape[0], 15, 3))
        features_rest_deg1 = features_extra[:, :3, :]
    elif len(extra_f_names) == 3 * 3:  # Already degree 1
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_rest_deg1 = features_extra.reshape((xyz.shape[0], 3, 3))
    else:
        features_rest_deg1 = np.zeros((xyz.shape[0], 3, 3))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # Filter by opacity threshold
    if opacity_threshold is not None and opacity_threshold > 0:
        opacity_mask = opacities.squeeze(-1) >= opacity_threshold
        xyz = xyz[opacity_mask]
        opacities = opacities[opacity_mask]
        features_dc = features_dc[opacity_mask]
        features_rest_deg1 = features_rest_deg1[opacity_mask]
        scales = scales[opacity_mask]
        rots = rots[opacity_mask]

    gs_params = {
        'means': torch.from_numpy(xyz).float(),
        'scales': torch.from_numpy(scales).float(),
        'opacities': torch.from_numpy(opacities).float(),
        'quats': torch.from_numpy(rots).float(),
        'features_dc': torch.from_numpy(features_dc).float().squeeze(-1),
        'features_rest': torch.from_numpy(features_rest_deg1).float(),
    }
    return gs_params


def preprocess_gaussian(gs_params, device, scaler_class=None, return_scaler=False):
    """Normalize gaussian parameters."""
    if scaler_class is None:
        from utils.transform_utils import MinMaxScaler
        scaler_class = MinMaxScaler
    
    scaler = scaler_class()
    gs_params_cuda = {k: v.to(device) for k, v in gs_params.items()}
    
    normalized_gs = {}
    normalized_gs['means'] = scaler.fit_transform(gs_params_cuda['means'])
    normalized_gs['scales'] = gs_params_cuda['scales'] + torch.log(scaler.scale_)
    normalized_gs['features_dc'] = gs_params_cuda['features_dc']
    normalized_gs['features_rest'] = gs_params_cuda['features_rest']
    normalized_gs['opacities'] = gs_params_cuda['opacities']
    normalized_gs['quats'] = gs_params_cuda['quats']
    
    if return_scaler:
        return normalized_gs, scaler
    return normalized_gs


def normalize_position(position, scaler, device):
    """Normalize a position using the same scaler used for gaussian means."""
    if not isinstance(position, torch.Tensor):
        position = torch.tensor(position, dtype=torch.float32, device=device)
    else:
        position = position.to(device=device, dtype=torch.float32)
    
    normalized_position = position * scaler.scale_ + scaler.trans_
    return normalized_position


def audio_path_to_ply_path(audio_path, ply_base):
    """Convert audio path to PLY path."""
    match = re.search(r'/(\d+)/\d+\.wav$', audio_path)
    if match:
        obj_id = match.group(1)
        return os.path.join(ply_base, obj_id, "model.ply")
    return None


class GaussianEncoder(nn.Module):
    """Gaussian Encoder based on PTv3."""
    
    def __init__(self, feature_predictor, output_dim=1024, output_seq_len=None, mode='generative'):
        super().__init__()
        self.feature_predictor = feature_predictor
        self.input_features = feature_predictor.input_features
        self.grid_resolution = feature_predictor.grid_resolution
        self.mode = mode
        self.output_seq_len = output_seq_len
        
        encoder_output_dim = 512
        
        if mode == 'contrastive':
            self.projection = nn.Sequential(
                nn.Linear(encoder_output_dim, encoder_output_dim),
                nn.GELU(),
                nn.Linear(encoder_output_dim, output_dim)
            )
        else:
            self.projection = nn.Linear(encoder_output_dim, output_dim)
    
    def _encode_ptv3(self, normalized_gs, device):
        """Run PTv3 encoder and return features."""
        from pointcept.models.utils.structure import Point
        ptv3 = self.feature_predictor.backbone.backbone
        
        offset = torch.tensor([normalized_gs['means'].shape[0]]).cumsum(0).to(device)
        feat_list = []
        for key in self.input_features:
            if key == 'means':
                feat_list.append(normalized_gs[key])
            elif key == 'features_rest':
                feat_list.append(normalized_gs[key].view(normalized_gs[key].shape[0], -1))
            else:
                feat_list.append(normalized_gs[key])
        feat = torch.cat(feat_list, dim=1)

        model_input = {
            'coord': normalized_gs['means'],
            'grid_size': torch.ones([3], device=device) * (1.0 / self.grid_resolution),
            'offset': offset,
            'feat': feat,
        }
        model_input['grid_coord'] = torch.floor(model_input['coord'] * self.grid_resolution).int()

        point = Point(model_input)
        point.serialization(order=ptv3.order, shuffle_orders=ptv3.shuffle_orders)
        point.sparsify()
        point = ptv3.embedding(point)
        point = ptv3.enc(point)
        
        return point.feat
    
    def forward(self, normalized_gs, device):
        """Forward pass."""
        encoder_feat = self._encode_ptv3(normalized_gs, device)
        
        if self.mode == 'contrastive':
            pooled_feat = encoder_feat.mean(dim=0, keepdim=True)
            return self.projection(pooled_feat)
        else:
            encoder_feat = self.projection(encoder_feat)
            n_points = encoder_feat.shape[0]
            if self.output_seq_len is not None:
                if n_points > self.output_seq_len:
                    indices = torch.linspace(0, n_points - 1, self.output_seq_len).long().to(device)
                    encoder_feat = encoder_feat[indices]
                elif n_points < self.output_seq_len:
                    padding = torch.zeros(self.output_seq_len - n_points, encoder_feat.shape[1], device=device)
                    encoder_feat = torch.cat([encoder_feat, padding], dim=0)
            return encoder_feat.unsqueeze(0)


def retrieve_timesteps(scheduler, num_inference_steps, device, timesteps=None, sigmas=None):
    """Retrieve timesteps for the scheduler."""
    if sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps
