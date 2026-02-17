# fixed_video_mae_for_pretraining.py
import torch
import torch.nn as nn
from transformers import VideoMAEConfig
from transformers import VideoMAEForPreTraining
from transformers import VideoMAEForVideoClassification

class CompatibleVideoMAEForPreTraining(nn.Module):
#    def __init__(self, pretrained_model_name="MCG-NJU/videomae-base", patch_dim=None):
    def __init__(self, pretrained_model_name="./videomae_base", patch_dim=None):
        super().__init__()
        
        # Load configuration and model from pretrained model
        self.config = VideoMAEConfig.from_pretrained(pretrained_model_name)
#        pretrained_model = VideoMAEForVideoClassification.from_pretrained(pretrained_model_name)
        pretrained_model = VideoMAEForPreTraining.from_pretrained(pretrained_model_name)
        self.videomae = pretrained_model.videomae
        
        # Get patch size
        self.patch_size = self._get_patch_size()
        
        # Dynamic calculation of patch dimension (ensure consistency between save and load)
        if patch_dim is None:
            self.patch_dim = self._calculate_patch_dim()
        else:
            self.patch_dim = patch_dim
        
        # Decoder configuration
        self.decoder_hidden_size = 512
        self.decoder_num_heads = 8
        self.decoder_num_layers = 4
        
        # Embedding layer for decoder
        self.decoder_embed = nn.Linear(self.config.hidden_size, self.decoder_hidden_size)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_hidden_size))
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.decoder_hidden_size,
            nhead=self.decoder_num_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.decoder_num_layers)
        
        # Prediction head (supports dynamic patch dimensions)
        self.decoder_pred = nn.Linear(self.decoder_hidden_size, self.patch_dim)
        
        # Initialization
        self._init_weights()
        
        print(f"VideoMAEForPreTraining initialization completed")
        print(f"Patch size: {self.patch_size}")
        print(f"Patch dimension: {self.patch_dim}")
    
    def _get_patch_size(self):
        """Safely get patch_size"""
        patch_size = getattr(self.config, 'patch_size', 16)
        
        if isinstance(patch_size, int):
            return [patch_size, patch_size]
        elif isinstance(patch_size, (list, tuple)):
            if len(patch_size) == 1:
                return [patch_size[0], patch_size[0]]
            else:
                return list(patch_size[:2])
        else:
            return [16, 16]
    
    def _calculate_patch_dim(self):
        """Calculate patch dimension"""
        patch_h, patch_w = self.patch_size
        tubelet_size = 2  # VideoMAE default
        channels = self.config.num_channels
        
        patch_dim = patch_h * patch_w * channels * tubelet_size
        return patch_dim
    
    def _init_weights(self):
        """Initialize weights of new layers"""
        torch.nn.init.xavier_uniform_(self.decoder_embed.weight)
        torch.nn.init.constant_(self.decoder_embed.bias, 0)
        torch.nn.init.xavier_uniform_(self.decoder_pred.weight)
        torch.nn.init.constant_(self.decoder_pred.bias, 0)
        torch.nn.init.normal_(self.mask_token, std=.02)
    
    def forward(self, pixel_values, bool_masked_pos=None):
        """Forward pass"""
        outputs = self.videomae(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        
        if bool_masked_pos is not None:
            batch_size, total_seq_len = bool_masked_pos.shape
            
            x_visible = self.decoder_embed(sequence_output)
            
            num_masked = bool_masked_pos.sum(dim=1)
            max_masked = num_masked.max().item()
            
            if max_masked > 0:
                mask_tokens = self.mask_token.expand(batch_size, max_masked, -1)
                
                x_full = torch.zeros(batch_size, total_seq_len, self.decoder_hidden_size, device=x_visible.device)
                
                for i in range(batch_size):
                    visible_mask = ~bool_masked_pos[i]
                    masked_mask = bool_masked_pos[i]
                    
                    x_full[i, visible_mask] = x_visible[i, :visible_mask.sum()]
                    
                    if masked_mask.sum() > 0:
                        x_full[i, masked_mask] = mask_tokens[i, :masked_mask.sum()]
                
                x = x_full
            else:
                x = x_visible
        else:
            x = self.decoder_embed(sequence_output)
        
        decoded = self.decoder(x, x)
        pred = self.decoder_pred(decoded)
        
        return pred

def load_compatible_model(model_path, device='cpu'):
    """Load compatible model"""
    
    print(f"Loading model: {model_path}")
    
    # First load checkpoint to check structure
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Estimate correct patch dimension from decoder_pred.weight shape
    decoder_pred_weight_shape = None
    for key, value in state_dict.items():
        if key == 'decoder_pred.weight':
            decoder_pred_weight_shape = value.shape
            break
    
    if decoder_pred_weight_shape is not None:
        patch_dim = decoder_pred_weight_shape[0]  # Output dimension
        print(f"Patch dimension estimated from checkpoint: {patch_dim}")
    else:
        patch_dim = None
        print("decoder_pred.weight not found. Using default patch dimension.")
    
    # Initialize model with correct patch dimension
    model = CompatibleVideoMAEForPreTraining(patch_dim=patch_dim)
    
    # Load state dictionary
    try:
        model.load_state_dict(state_dict)
        print("✓ Model loading successful")
    except RuntimeError as e:
        print(f"Warning: Complete loading failed: {e}")
        # Attempt partial loading
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"✓ Partial loading completed ({len(pretrained_dict)}/{len(state_dict)} layers)")
    
    return model

