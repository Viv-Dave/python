import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import math

from transformers import Blip2Processor, Blip2Model, Blip2Config

device = "cuda" if torch.cuda.is_available() else "cpu"

#BLIP Hyperparameters
"""
hidden_size =768
num_attn_heads = 12
intermediate_size = 3072
dropout = 0.1
"""
hidden_size =384
num_attn_heads = 8
intermediate_size = 1536
dropout = 0.1
actual_size = 768
batch_size = 4
num_queries = 32         
query_hidden_dim = 768   

num_image_patches = 257  
vision_hidden_dim = 1408

class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=True)  # gate
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=True)  # value
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.w1(x)) * self.w2(x)
    
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        self.register_buffer("cos", freqs.cos()) 
        self.register_buffer("sin", freqs.sin())  

    def apply_rotary(self, x, seq_dim=2):
        """
        x: [B, H, seq, D]
        """
        cos = self.cos[: x.size(seq_dim)].unsqueeze(0).unsqueeze(0)  # [1,1,seq,d/2]
        sin = self.sin[: x.size(seq_dim)].unsqueeze(0).unsqueeze(0)
        
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        rotated = torch.cat([x1 * cos - x2 * sin,
                             x1 * sin + x2 * cos], dim=-1)
        return rotated

class BLIP2MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size=hidden_size, num_attention_heads=num_attn_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Query/Key/Value projections
        self.query = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.key   = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=True)

        # Output projection
        self.out_proj = nn.Linear(self.all_head_size, hidden_size, bias=True)

        self.dropout = nn.Dropout(dropout, inplace=False)

        # ---- Add RoPE ----
        self.rope = RotaryEmbedding(self.attention_head_size)

    def transpose_for_scores(self, x):
        # [B, seq, hidden] → [B, heads, seq, head_dim]
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer   = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            key_layer   = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

            if past_key_value is not None:
                key_layer   = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)


        query_layer = self.transpose_for_scores(self.query(hidden_states))

        query_layer = self.rope.apply_rotary(query_layer)
        key_layer   = self.rope.apply_rotary(key_layer)


        present_key_value = (key_layer, value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs_dropped = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(*new_shape)
        context_layer = self.out_proj(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = outputs + (present_key_value,)
        return outputs

class BLIP2QFormerOutput(nn.Module):
    def __init__(self, hidden_size=hidden_size, intermediate_size=intermediate_size):
        super(BLIP2QFormerOutput, self).__init__()
        # FFN output projects intermediate->hidden
        self.dense = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout, inplace=False)
        # self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12, elementwise_affine=True)
        self.RMSNorm = nn.RMSNorm(hidden_size, eps=1e-12, elementwise_affine=True)
    def forward(self, hidden_states, input_tensor):
        # hidden_states: [B, seq, intermediate_size]  (FFN output)
        hidden_states = self.dense(hidden_states)    # 3072 -> 768
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.RMSNorm(hidden_states + input_tensor)
        return hidden_states

class BLIP2QFormerSelfOutput(nn.Module):
    def __init__(self, hidden_size=hidden_size):
        super(BLIP2QFormerSelfOutput, self).__init__()
        # Attention output projects hidden->hidden
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout, inplace=False)
        # self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12, elementwise_affine=True)
        self.RMSNorm = nn.RMSNorm(hidden_size, eps=1e-12, elementwise_affine=True)
    def forward(self, hidden_states, input_tensor):
        # hidden_states: [B, seq, hidden_size]  (attention context)
        hidden_states = self.dense(hidden_states)    # 768 -> 768
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.RMSNorm(hidden_states + input_tensor)
        return hidden_states

class BLIP2QFormerAttention(nn.Module):
    def __init__(self, hidden_size=hidden_size, num_attention_heads=num_attn_heads):
        super(BLIP2QFormerAttention, self).__init__()
        self.attention = BLIP2MultiHeadAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads)
        self.output = BLIP2QFormerSelfOutput(hidden_size=hidden_size)
        self.dropout = nn.Dropout(dropout, inplace=False)
        # self.layernorm = nn.LayerNorm(hidden_size, eps=1e-12, elementwise_affine=True)
        # self.RMSNorm = nn.RMSNorm(hidden_size, eps=1e-12, elementwise_affine=True)
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # Forward through MHA
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

        attention_output = self_outputs[0]
        # Post-attention dense + residual + layernorm is handled in SelfOutput (keeps your style)
        attention_output = self.output(attention_output, hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # keep attentions & present_key_value if requested
        return outputs

class BLIP2QFormerIntermediate(nn.Module):
    def __init__(self, hidden_size=hidden_size, intermediate_size=intermediate_size):
        super().__init__()
        # No dense before SwiGLU
        self.act_fn = SwiGLU(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        return self.act_fn(hidden_states)

class QFormerLayer(nn.Module):
    def __init__(self, hidden_size=hidden_size, num_attention_heads=num_attn_heads, intermediate_size=intermediate_size, actual_size=actual_size):
        super(QFormerLayer, self).__init__()
        self.attention = BLIP2QFormerAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads)
        self.crossattention = BLIP2QFormerAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads)

        self.intermediate_query = BLIP2QFormerIntermediate(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.output_query = BLIP2QFormerOutput(
                                hidden_size=hidden_size,
                                intermediate_size=intermediate_size,
                            )
        
    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        # 1) Self-attention (Now handles Text+Query masking)
        self_attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask # <--- NEW: Pass the mask down
        )
        attention_output = self_attention_outputs[0]

        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                hidden_states=attention_output,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            
            attention_output = cross_attention_outputs[0]

        # 3) FFN
        intermediate_states = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_states, attention_output)
        return (layer_output,)

class QFormerModel(nn.Module):
    def __init__(self, num_layers=4, hidden_size=384, num_heads=num_attn_heads):
        super().__init__()
        
        # FIX: Calculate intermediate size (standard is 4x hidden) or use global
        # Using 4x hidden here to be safe, or you can use your global 'intermediate_size'
        inter_size = hidden_size * 4 

        # FIX: Pass all required arguments to QFormerLayer
        self.layers = nn.ModuleList([
            QFormerLayer(
                hidden_size=hidden_size, 
                num_attention_heads=num_heads, 
                intermediate_size=inter_size
            ) 
            for _ in range(num_layers)
        ])
        
        self.text_input_proj   = nn.Linear(768, hidden_size)
        self.image_input_proj  = nn.Linear(1408, hidden_size)
        self.output_proj       = nn.Linear(hidden_size, 768)
        self.query_tokens = nn.Parameter(torch.randn(1, 32, 384)) 
        
    def forward(self, encoder_hidden_states=None, encoder_attention_mask=None, text_hidden_states=None, text_attention_mask=None):
        """
            encoder_hidden_states  : [B, 257, 1408] (Original ViT features)
            text_hidden_states     : [B, Seq, 768]  (Optional BERT embeddings)
            text_attention_mask    : [B, Seq]       (Optional Mask)
         """
        B = encoder_hidden_states.size(0)
        
        # 1. Expand Query Tokens
        query_embeds = self.query_tokens.expand(B, -1, -1) # [B, 32, 384]

        # 2. Handle Text Input (Concatenation)
        if text_hidden_states is not None:
            # Project text to match Q-Former dimension
            text_embeds = self.text_input_proj(text_hidden_states)
            
            # Concatenate [Queries, Text]
            hidden_states = torch.cat([query_embeds, text_embeds], dim=1)
            
            # Create Combined Attention Mask
            # Queries are always visible (1), Text depends on input mask
            query_mask = torch.ones(B, 32, device=hidden_states.device)
            if text_attention_mask is None:
                text_attention_mask = torch.ones(B, text_embeds.shape[1], device=hidden_states.device)
                
            attention_mask = torch.cat([query_mask, text_attention_mask], dim=1)
            
            # Convert 1/0 mask to additive mask (0.0 for keep, -10000.0 for mask) for Softmax
            # Reshape from [B, Seq] to [B, 1, 1, Seq] for MultiHeadAttention
            attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
        else:
            hidden_states = query_embeds
            attention_mask = None

        # 3. Handle Image Input
        if encoder_hidden_states is not None:
            encoder_hidden_states = self.image_input_proj(encoder_hidden_states)

        # 4. Pass through Layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, 
                attention_mask=attention_mask, # Pass the mask
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )[0]

        # 5. Output Projection
        # IMPORTANT: We only return the updated Query Tokens (first 32), ignoring the concatenated text tokens
        query_outputs = hidden_states[:, :32, :]
        
        output = self.output_proj(query_outputs)
        return (output,)
if __name__ == "__main__":

    config = Blip2Config() # This will use the default dimensions
    my_qformer_layer = QFormerModel()
    my_qformer_layer.eval() # Set to evaluation mode