
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class LearnableTextPrompt(nn.Module):

    def __init__(self, clip_model, class_names, n_ctx=4):

        super().__init__()

        self.clip_model = clip_model
        self.class_names = class_names
        self.n_ctx = n_ctx

        ctx_dim = clip_model.token_embedding.weight.shape[1]
        device = clip_model.token_embedding.weight.device

        # Learnable context tokens
        self.context = nn.Parameter(
            torch.randn(n_ctx, ctx_dim, device=device) * 0.02
        )

        # Reserve positions for context tokens
        prompts = [
            f"a video of {' '.join(['X'] * n_ctx)} {name}"
            for name in class_names
        ]

        tokenized_prompts = open_clip.tokenize(prompts).to(device)

        prefix_tokens = open_clip.tokenize(
            ["a video of"]
        ).to(device)

        # Find the position immediately after the fixed prefix
        prefix_length = prefix_tokens.argmax(dim=-1).item()

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)

        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("prefix", embedding[:, :prefix_length, :])
        self.register_buffer(
            "suffix",
            embedding[:, prefix_length + n_ctx:, :]
        )

    def forward(self):

        batch_size = len(self.class_names)

        context = self.context.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # Preserve class names after the learnable context
        prompts = torch.cat(
            [self.prefix, context, self.suffix],
            dim=1
        )

        prompts = prompts + self.clip_model.positional_embedding

        x = self.clip_model.transformer(
            prompts,
            attn_mask=self.clip_model.attn_mask
        )

        x = self.clip_model.ln_final(x)

        eot_positions = self.tokenized_prompts.argmax(dim=-1)

        x = x[
            torch.arange(batch_size, device=x.device),
            eot_positions
        ]

        x = x @ self.clip_model.text_projection

        return F.normalize(x, dim=-1)
