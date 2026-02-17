import torch
import torch.nn as nn
from torchvision.ops import MLP


class VideomaePooling(nn.Module):
    def __init__(self,
                 general_cfg,   # General configuration 
                 mlp_cfg):      # Configuration for the MLP
        super().__init__()

        # General configurations
        self.approach = general_cfg.approach
        self.feature_dim = general_cfg.dim

        # MLP for final output
        if 'norm_layer' in mlp_cfg.keys():
            mlp_cfg['norm_layer'] = eval(mlp_cfg['norm_layer'])
#            mlp_cfg['norm_layer'] = mlp_cfg['norm_layer']
        if 'activation_layer' in mlp_cfg.keys():
            mlp_cfg['activation_layer'] = eval(mlp_cfg['activation_layer'])
#            mlp_cfg['activation_layer'] = mlp_cfg['activation_layer']

        if self.approach == 'global_pooling':
            self.mlp = self._build_global_pooling_classifier(mlp_cfg)
        elif self.approach == 'attention_pooling':
            self.attention, self.mlp = self._build_attention_pooling_classifier(mlp_cfg)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")
    
    def _build_global_pooling_classifier(self, mlp_cfg):
        """ global pooling """
        mlp = MLP(**mlp_cfg)

        return mlp
    
    def _build_attention_pooling_classifier(self, mlp_cfg):
        """ attention pooling """
        attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            batch_first=True
        )

        mlp = MLP(**mlp_cfg)

        return attention, mlp
    
    def forward(self, features):
        """
        Args:
            features: (batch_size, seq_len, feature_dim) = (B, 90, 768)
        Returns:
            logits: (batch_size, num_classes)
        """
        
        if self.approach == 'global_pooling':
            # global pooling
            pooled_features = features.mean(dim=1)  # (B, 768)
            y = self.mlp(pooled_features)
            return y
        
        elif self.approach == 'attention_pooling':
            # attention pooling. use the average feature as the query
            query = features.mean(dim=1, keepdim=True)  # (B, 1, 768)
            attended_features, _ = self.attention(query, features, features)  # (B, 1, 768)
            pooled_features = attended_features.squeeze(1)  # (B, 768)
            y = self.mlp(pooled_features)
            return y

if __name__ == "__main__":
    from read_yaml import read_yaml

#    cfg = read_yaml('yaml/videomae_global_pooling_grs_cr.yaml')
    cfg = read_yaml('yaml/videomae_attention_pooling_grs_cr.yaml')

    model = VideomaePooling(cfg.Model.params.general,
                            cfg.Model.params.mlp).to('cuda')

    # 8 clips: [B, 38, 1408]
    clip_features = torch.randn(8, 90, 768).to('cuda')
    out = model(clip_features)  # → [1, 768]
    print("Output shape:", out.shape)  # [1, 768]