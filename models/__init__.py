import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from .latte import Latte_models
from .latte_t2v import LatteT2V

from torch.optim.lr_scheduler import LambdaLR


def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)
    
def get_models(args):
    if 'LaHT' in args.model:
        # TODO: use community weights
        return LatteT2V(num_attention_heads=args.num_attention_heads, 
                        attention_head_dim=args.attention_head_dim,
                        in_channels=4,
                        patch_size=args.patch_size,
                        sample_size=32,
                        caption_channels=args.caption_channels,
                        cross_attention_dim=args.cross_attention_dim,
                        norm_type='ada_norm_single',
                        num_layers=args.num_layer,
                        video_length=args.num_frames,
                        efficient_mode=args.enable_xformers_memory_efficient_attention,
                        gradient_checkpointing=args.gradient_checkpointing,
                        align_camera=args.align_camera,
                        camera_dim=args.camera_dim)
    elif 'Lat-H' in args.model:
        return LatteT2V.from_pretrained(args.pretrained_model_path, subfolder="transformer", video_length=args.video_length, low_cpu_mem_usage=False)
    elif 'LaH' in args.model:
        return Latte_models[args.model](
                input_size=args.latent_size,
                num_classes=args.num_classes,
                num_frames=args.num_frames,
                learn_sigma=args.learn_sigma,
                extras=args.extras
            )
    else:
        raise '{} Model Not Supported!'.format(args.model)
    