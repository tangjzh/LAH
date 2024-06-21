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
        return LatteT2V(num_attention_heads=16, 
                        in_channels=4,
                        patch_size=2,
                        sample_size=32,
                        caption_channels=768,
                        cross_attention_dim=1408,
                        norm_type='ada_norm_single',
                        num_layers=4,
                        video_length=args.video_length)
        # return LatteT2V.from_pretrained(args.pretrained_model_path, subfolder="transformer", video_length=args.video_length)
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
    