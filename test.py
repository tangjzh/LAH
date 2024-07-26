import torch
from models.latte_t2v import LatteT2V

def test_LatteT2V():
    # 定义模型参数
    num_attention_heads = 16
    attention_head_dim = 72
    in_channels = 4
    out_channels = 3
    num_layers = 16
    dropout = 0.1
    cross_attention_dim = 1152
    sample_size = 32
    patch_size = 2
    num_embeds_ada_norm = None
    use_linear_projection = False
    only_cross_attention = False
    double_self_attention = False
    upcast_attention = False
    norm_type = 'ada_norm_single'
    norm_elementwise_affine = True
    norm_eps = 1e-5
    attention_type = "default"
    caption_channels = 768
    video_length = 8
    camera_dim = 128
    
    # 初始化模型
    model = LatteT2V(
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        in_channels=in_channels,
        num_layers=num_layers,
        cross_attention_dim=cross_attention_dim,
        sample_size=sample_size,
        patch_size=patch_size,
        norm_type=norm_type,
        caption_channels=caption_channels,
        video_length=video_length,
        camera_dim=camera_dim
    )
    
    # 创建一个假输入
    batch_size = 2
    num_frames = 8
    height = sample_size
    width = sample_size
    hidden_states = torch.randn(batch_size, num_frames, in_channels, height, width)
    encoder_hidden_states = torch.randn(batch_size, 120, 768)
    timestep = torch.randint(0, 1000, (batch_size, ))
    camera_pose = torch.randn(batch_size, num_frames, 3)
    camera_ray = torch.randn(batch_size, num_frames, 256, 256, 3)
    
    # 运行模型
    output = model(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        camera_pose=camera_pose,
        camera_ray=camera_ray,
        enable_time=False
    )

    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"Parameter {name} did not receive grad update for rank 0.")
    
    # 打印输出的形状
    print("Output shape:", output.shape)

# 运行测试
test_LatteT2V()
