import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange
import torchvision
import utils
import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import fusion


from torch.nn.functional import normalize
from basic_blocks import *

UCF_SAMPLE_RATE = 44100
CLAP_DURATION = 7


####################################
#                                  #
#                                  #
#   TempNet Architecture           #
#                                  #
#                                  #
####################################
   
class TransformerNet(nn.Module):
    def __init__(self, mode='fusion_audio', video_frames_per_clip=16, audio_frames_per_clip=16,
                 emb_size=512, audio_frame=689, num_mel=224, num_class=51,
                 num_self_attn_layers=2, num_cross_attn_layers=2, num_heads=4):
        super(TransformerNet, self).__init__()
        self.mode = mode # 'video', 'audio', 'fusion_video', 'fusion_audio'
        self.num_cross_attn_layers = num_cross_attn_layers

        # video
        self.image_encoder = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        # freeze resnet
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.image_encoder_fc = nn.Linear(1000, emb_size)
        self.video_pos_emb = PositionalEncoding(len=video_frames_per_clip, embed_dim=emb_size)
        self.video_self_attn = nn.ModuleList([SelfAttentionEncoder(emb_size, num_heads) for _ in range(num_self_attn_layers)])
        self.video_crs_attn = nn.ModuleList([CrossAttentionEncoder(emb_size, num_heads) for _ in range(num_cross_attn_layers)])

        # audio
        self.audio_mel_proj = nn.Linear(num_mel, emb_size)
        # self.audio_time_conv = nn.Conv1d(in_channels=audio_frame, out_channels=frames_per_clip, kernel_size=1)
        self.audio_time_proj = nn.Sequential(
            nn.Linear(audio_frame, audio_frames_per_clip+(audio_frame-audio_frames_per_clip)//2),
            nn.ReLU(),
            nn.LayerNorm(audio_frames_per_clip+(audio_frame-audio_frames_per_clip)//2),
            nn.Linear(audio_frames_per_clip+(audio_frame-audio_frames_per_clip)//2, audio_frames_per_clip)
        )
        # self.audio_pos_emb = PositionalEncoding(len=frames_per_clip, embed_dim=emb_size)
        self.audio_pos_emb = PositionalEncoding(len=audio_frames_per_clip, embed_dim=emb_size)
        self.audio_self_attn = nn.ModuleList([SelfAttentionEncoder(emb_size, num_heads) for _ in range(num_self_attn_layers)])
        self.audio_crs_attn = nn.ModuleList([CrossAttentionEncoder(emb_size, num_heads) for _ in range(num_cross_attn_layers)])

        # classification
        self.fc = nn.Linear(emb_size, num_class)

    def forward(self, video, audio):
        # video shape [b, t, 3, 224, 224]
        # audio shape [b, audio_frame, num_mel]

        # video representation
        if self.mode != 'audio':
            # combine b, t dim
            b, tv, c, h, w = video.shape

            video = rearrange(video, 'b t c h w -> (b t) c h w')

            video = self.image_encoder(video) # exp ((b, tv), emb_size)
            video = self.image_encoder_fc(video)

            # seperate b, t dim back
            video = rearrange(video, '(b t) d -> b t d', b=b) # exp (b, tv, emb_size)

            # temporal positional embedding
            video = self.video_pos_emb(video)

            # temporal self atten
            for layer in self.video_self_attn:
                video = layer(video, mask=None) # exp (b, tv, emb_size)

        # audio representation
        if self.mode != 'video':
            # linear proj to emb size
            audio = self.audio_mel_proj(audio) # exp (b, audio_frame, emb_size)

            # conv to time
            # audio = self.audio_time_conv(audio) # exp (b, t, emb_size)

            # proj to time
            audio = rearrange(audio, 'b x d -> b d x') # exp (b, emb_size, tx)
            audio = self.audio_time_proj(audio) # exp (b, emb_size, ta)
            audio = rearrange(audio, 'b d t -> b t d') # exp (b, ta, emb_size)

            # temporal positional embedding
            audio = self.audio_pos_emb(audio)

            # temporal self attn
            for layer in self.audio_self_attn:
                audio = layer(audio, mask=None) # exp (b, ta, emb_size)

        # cross-modal attn
        if self.mode == 'fusion_audio' or self.mode == 'fusion_video':
            for i in range(self.num_cross_attn_layers):
                video = self.video_crs_attn[i](video, audio, mask=None) # exp (b, ta, emb_size)
                audio = self.audio_crs_attn[i](video, audio, mask=None) # exp (b, ta, emb_size)

        if self.mode == 'audio' or self.mode == 'fusion_audio':
            feat = audio
        else: feat = video

        # temporal pooling
        feat = nn.functional.avg_pool1d(feat.permute(0, 2, 1), kernel_size=feat.shape[1]).squeeze(-1) # exp (b, emb_size)
        logits = self.fc(feat) # exp (b, num_class)

        return logits, feat

class TempNet(nn.Module):
    def __init__(self, videomae_model, text_features, av_emb_size=768, device=torch.device("cuda")):
        super(TempNet, self).__init__()

        self.videomae_model = videomae_model
        self.device= device
        self.text_features = text_features

        # classification
        self.fc = nn.Sequential(
            nn.Linear(av_emb_size, text_features.shape[-1]),
            # nn.ReLU(),
            # nn.LayerNorm(av_emb_size//2),
            # nn.Linear(av_emb_size//2, num_class)
        )
        

    def forward(self, batch):
        video_feat = utils.get_videomae_feats(self.videomae_model, batch, self.device, freeze=True) # (b, 1568, 768)

        # pooling
        av_feat = nn.functional.avg_pool1d(video_feat.permute(0, 2, 1), kernel_size=video_feat.shape[1]).squeeze(-1) # exp (b, av_emb_size)
        av_feat = self.fc(av_feat) # exp (b, text_emb_size)

        av_features = av_feat / av_feat.norm(dim=-1, keepdim=True)

        logits_per_av = 99.8748 * av_features @ self.text_features.T # logit_scale from ViFi-CLIP, CLIP used 100
        logits_per_text = 99.8748 * self.text_features @ av_features.T

        return logits_per_av, logits_per_text, av_features, self.text_features
    


####################################
#                                  #
#                                  #
#   VCLAPNet Architecture          #
#                                  #
#                                  #
####################################



class VCLAPNet(nn.Module):
    def __init__(self, videomae_model, classnames, clip_model, clap_model, device, use_videomae=False, use_audio=True):
        super().__init__()
        self.prompt_learner = VLPromptLearner(classnames, clip_model, device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        if use_videomae:
            print("Using VideoMae model for image encoder")
            self.image_encoder = videomae_model
        else:
            print("Using CLIP model for image encoder")
            self.image_encoder = clip_model.visual
        self.use_videomae = use_videomae

        self.audio_encoder = clap_model
        
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device
        self.use_audio = use_audio
        
        self.fc = nn.Sequential(
            nn.Linear(768, 512), #might be a problem
        )
        self.fc_clap = nn.Sequential(
            nn.Linear(1024, 768), #might be a problem
        )

        self.fusion = fusion.SummationFusion(dim1=768, dim2=1024, fuse_dim=768)
        

    def forward(self, batch):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        prompts = self.prompt_learner()
                
        if self.use_videomae:
            # # Now pass the image into CLIP visual encoder
            video_feat = utils.get_videomae_feats(self.image_encoder, batch, self.device, freeze=True) #torch.Size([8, 1568, 768])
            video_feat = nn.functional.avg_pool1d(video_feat.permute(0, 2, 1), kernel_size=video_feat.shape[1]).squeeze(-1) # exp (b, av_emb_size) torch.Size([8, 768])
            video_feat = self.fc(video_feat) #torch.Size([8, 512])
        else:
            image = batch['pixel_values']
            b, t, c, h, w = image.size()
            # Remove the batch dimensions
            image = image.reshape(-1, c, h, w)
            # Now pass the image into CLIP visual encoder
            image_features = self.image_encoder(image.type(self.dtype))
            # Now again attach the batch dimensions
            image_features = image_features.view(b, t, -1)  # [B, T, 512]
            # Now take the mean along the temporal direction
            video_feat = image_features.mean(dim=1, keepdim=False)  # image features are now ready

        # audio feat
        audio = utils.load_batch_audio_into_tensor(batch["audio"], UCF_SAMPLE_RATE, CLAP_DURATION).to(self.device) # sample rate and duration are both hardcoded
        audio_feat = self.audio_encoder._get_audio_embeddings(audio) #(b, 1024)

        # audio-visual fusion
        if self.use_audio:
            av_feat = self.fusion(video_feat, audio_feat)
        else: av_feat = video_feat


        # Finally, make the text features
        text_feat = self.text_encoder(prompts, tokenized_prompts)

        av_feat = av_feat / av_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        logits = logit_scale * av_feat @ text_feat.t().float()
    

        return logits, logits.t(), av_feat, text_feat



####################################
#                                  #
#                                  #
#   ViFiCLIP Architecture          #
#                                  #
#                                  #
####################################



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, device):
        super().__init__()
        dtype = clip_model.dtype
        self.use_prompt_stage = False # second stage prompting?
        ctx_init = "a video of a"  # initialization words (only for language prompts)
        ZS_evaluation = False
        self.PROMPT_DEPTH_TEXT = 9 # max 12, min 0, for 0 it will act as shallow language prompting (first layer)
        self.PROMPT_DEPTH_VISION = 9 # max 12, min 0, for 0 it will act as shallow vision prompting (first layer)
        self.N_CTX_TEXT = 16 # number of context vectors at the language branch
        self.N_CTX_VISION = 16  # number of context vectors at the vision branch
        
        if ZS_evaluation:
            text_aug = f"{{}}"
            tokenized_prompts = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for c in classnames]).to(device)
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).cuda()
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        elif self.use_prompt_stage:
            n_cls = len(classnames)
            # Make sure Language depth >= 1
            assert self.PROMPT_DEPTH_TEXT >= 1, "In VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
            n_ctx = self.N_CTX_TEXT
            ctx_dim = clip_model.ln_final.weight.shape[0]

            if ctx_init and (n_ctx) <= 4:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = n_ctx
                prompt = clip.tokenize(ctx_init).to(device)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                prompt_prefix = ctx_init
            else:
                # random initialization
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
            print(f"V-L design")
            print(f'Initial text context: "{prompt_prefix}"')
            print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
            print(f"Number of context words (tokens) for Vision prompting: {self.N_CTX_VISION}")
            self.ctx = nn.Parameter(ctx_vectors)

            classnames = [name.replace("_", " ") for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]

            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use
            # those computed using the current class names
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            self.n_cls = n_cls
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        else:
            # No prompting
            ctx_init = ctx_init.replace("_", " ")
            prompt_prefix = ctx_init
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        if self.use_prompt_stage:
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix
            prompts = self.construct_prompts(ctx, prefix, suffix)
        else:
            prompts = self.complete_text_embeddings

        return prompts

class AlignNet(nn.Module):
    def __init__(self, videomae_model, classnames, clip_model, device, use_videomae=False):
        super().__init__()
        self.prompt_learner = VLPromptLearner(classnames, clip_model, device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        if use_videomae:
            print("Using VideoMae model for image encoder")
            self.image_encoder = videomae_model
        else:
            print("Using CLIP model for image encoder")
            self.image_encoder = clip_model.visual
        self.use_videomae = use_videomae
        
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device
        self.visual_freeze, self.text_freeze = False, False
        
        self.fc = nn.Sequential(
            nn.Linear(768, 512), #might be a problem
        )
        

    def forward(self, batch):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        prompts = self.prompt_learner()
                
        if self.use_videomae:
            # # Now pass the image into CLIP visual encoder
            video_feat = utils.get_videomae_feats(self.image_encoder, batch, self.device, freeze=self.visual_freeze) #torch.Size([8, 1568, 768])
            video_feat = nn.functional.avg_pool1d(video_feat.permute(0, 2, 1), kernel_size=video_feat.shape[1]).squeeze(-1) # exp (b, av_emb_size) torch.Size([8, 768])
            #video_feat = self.fc(video_feat) #torch.Size([8, 512])
        else:
            image = batch['pixel_values']
            b, t, c, h, w = image.size()
            # Remove the batch dimensions
            image = image.reshape(-1, c, h, w)
            # Now pass the image into CLIP visual encoder
            image_features = self.image_encoder(image.type(self.dtype))
            # Now again attach the batch dimensions
            image_features = image_features.view(b, t, -1)  # [B, T, 512]
            # Now take the mean along the temporal direction
            video_feat = image_features.mean(dim=1, keepdim=False)  # image features are now ready


        # Finally, make the text features
        text_feat = self.text_encoder(prompts, tokenized_prompts)

        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        logits = logit_scale * video_feat @ text_feat.t().float()
    


        return logits, logits.t(), video_feat, text_feat
    
    
    def freeze(self, visual = True, text = True):
        
        if visual:
            utils.freeze(self.image_encoder)
            print("Visual encoder freezed")
            self.visual_freeze = True
        if text:
            utils.freeze(self.text_encoder)
            print("Text encoder freezed")
            self.text_freeze = True
            
        
