import torch
import torch.nn as nn
import torch.nn.functional as F

## The Truncation Trick and Orthogonal Regularization
def orthogonal_regularization(weight):
    '''
    Function for computing the orthogonal regularization term for a given weight matrix.
    '''
    weight = weight.flatten(1)
    return torch.norm(
        torch.dot(weight, weight) * (torch.ones_like(weight) - torch.eye(weight.shape[0]))
    )

##Class-conditional Batch Normalization
class ClassConditionalBatchNorm2d(nn.Module):
    '''
    ClassConditionalBatchNorm2d Class
    Values:
    in_channels: the dimension of the class embedding (c) + noise vector (z), a scalar
    out_channels: the dimension of the activation tensor to be normalized, a scalar
    '''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.class_scale_transform = nn.utils.spectral_norm(nn.Linear(in_channels, out_channels, bias=False))
        self.class_shift_transform = nn.utils.spectral_norm(nn.Linear(in_channels, out_channels, bias=False))

    def forward(self, x, y):
        normalized_image = self.bn(x)
        class_scale = (1 + self.class_scale_transform(y))[:, :, None, None]
        class_shift = self.class_shift_transform(y)[:, :, None, None]
        transformed_image = class_scale * normalized_image + class_shift
        return transformed_image

# class AdaIN(nn.Module):
#     '''
#     AdaIN Class, extends/subclass of nn.Module
#     Values:
#       channels: the number of channels the image has, a scalar
#       w_dim: the dimension of the intermediate tensor, w, a scalar 
#     '''

#     def __init__(self, channels, w_dim):
#         super().__init__()
#         self.instance_norm = nn.InstanceNorm2d(channels)
#         self.style_scale_transform = nn.Linear(w_dim, channels)
#         self.style_shift_transform = nn.Linear(w_dim, channels)

#     def forward(self, image, w):
#         normalized_image = self.instance_norm(image)
#         style_scale = self.style_scale_transform(w)[:, :, None, None]
#         style_shift = self.style_shift_transform(w)[:, :, None, None]
#         transformed_image = style_scale * normalized_image + style_shift
#         return transformed_image

##Self-Attention
class AttentionBlock(nn.Module):
    '''
    AttentionBlock Class
    Values:
    channels: number of channels in input
    '''
    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        self.theta = nn.utils.spectral_norm(nn.Conv2d(channels, channels // 8, kernel_size=1, padding=0, bias=False))
        self.phi = nn.utils.spectral_norm(nn.Conv2d(channels, channels // 8, kernel_size=1, padding=0, bias=False))
        self.g = nn.utils.spectral_norm(nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=False))
        self.o = nn.utils.spectral_norm(nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0, bias=False))

        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        spatial_size = x.shape[2] * x.shape[3]

        # Apply convolutions to get query (theta), key (phi), and value (g) transforms
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), kernel_size=2)
        g = F.max_pool2d(self.g(x), kernel_size=2)

        # Reshape spatial size for self-attention
        theta = theta.view(-1, self.channels // 8, spatial_size)
        phi = phi.view(-1, self.channels // 8, spatial_size // 4)
        g = g.view(-1, self.channels // 2, spatial_size // 4)

        # Compute dot product attention with query (theta) and key (phi) matrices
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), dim=-1)

        # Compute scaled dot product attention with value (g) and attention (beta) matrices
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.channels // 2, x.shape[2], x.shape[3]))

        # Apply gain and residual
        return self.gamma * o + x

## Generator
class GResidualBlock(nn.Module):
    '''
    GResidualBlock Class
    Values:
    c_dim: the dimension of conditional vector [c, z], a scalar
    in_channels: the number of channels in the input, a scalar
    out_channels: the number of channels in the output, a scalar
    '''

    def __init__(self, c_dim, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        self.bn1 = ClassConditionalBatchNorm2d(c_dim, in_channels)
        self.bn2 = ClassConditionalBatchNorm2d(c_dim, out_channels)

        self.activation = nn.ReLU()
        self.upsample_fn = nn.Upsample(scale_factor=2)     # upsample occurs in every gblock

        self.mixin = (in_channels != out_channels)
        if self.mixin:
            self.conv_mixin = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))

    def forward(self, x, y):
        # h := upsample(x, y)
        h = self.bn1(x, y)
        h = self.activation(h)
        h = self.upsample_fn(h)
        h = self.conv1(h)

        # h := conv(h, y)
        h = self.bn2(h, y)
        h = self.activation(h)
        h = self.conv2(h)

        # x := upsample(x)
        x = self.upsample_fn(x)
        if self.mixin:
            x = self.conv_mixin(x)

        return h + x

class Generator(nn.Module):
    '''
    Generator Class
    Values:
    z_dim: the dimension of random noise sampled, a scalar
    shared_dim: the dimension of shared class embeddings, a scalar
    base_channels: the number of base channels, a scalar
    bottom_width: the height/width of image before it gets upsampled, a scalar
    n_classes: the number of image classes, a scalar
    '''
 
    def __init__(self, base_channels=96, bottom_width=4, z_dim=120, shared_dim=128, n_classes=1000):
        super().__init__()
 
        n_chunks = 6    # 5 (generator blocks) + 1 (generator input)
        self.z_chunk_size = z_dim // n_chunks
        self.z_dim = z_dim
        self.shared_dim = shared_dim
        self.bottom_width = bottom_width
 
        # No spectral normalization on embeddings, which authors observe to cripple the generator
        self.shared_emb = nn.Embedding(n_classes, shared_dim)
 
        self.proj_z = nn.Linear(self.z_chunk_size, 16 * base_channels * bottom_width ** 2)
 
        # Can't use one big nn.Sequential since we are adding class+noise at each block
        self.g_blocks = nn.ModuleList([
            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 16 * base_channels, 16 * base_channels),
                AttentionBlock(16 * base_channels),
            ]),
            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 16 * base_channels, 8 * base_channels),
                AttentionBlock(8 * base_channels),
            ]),
            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 8 * base_channels, 4 * base_channels),
                AttentionBlock(4 * base_channels),
            ]),
            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 4 * base_channels, 2 * base_channels),
                AttentionBlock(2 * base_channels),
            ]),
            nn.ModuleList([
                GResidualBlock(shared_dim + self.z_chunk_size, 2 * base_channels, base_channels),
                AttentionBlock(base_channels),
            ]),
        ])
        self.proj_o = nn.Sequential(
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(base_channels, 3, kernel_size=1, padding=0)),
            nn.Tanh(),
        )
 
    def forward(self, z, y):
        '''
        z: random noise with size self.z_dim
        y: class embeddings with size self.shared_dim
            = NOTE =
            y should be class embeddings from self.shared_emb, not the raw class labels
        '''
        # Chunk z and concatenate to shared class embeddings
        zs = torch.split(z, self.z_chunk_size, dim=1)
        z = zs[0]
        ys = [torch.cat([y, z], dim=1) for z in zs[1:]]
 
        # Project noise and reshape to feed through generator blocks
        h = self.proj_z(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
 
        # Feed through generator blocks
        for idx, g_block in enumerate(self.g_blocks):
            h = g_block[0](h, ys[idx])
            h = g_block[1](h)
 
        # Project to 3 RGB channels with tanh to map values to [-1, 1]
        h = self.proj_o(h)
 
        return h


###'''BigGAN Discriminator
###Before implementing the discriminator in full, you need to implement a discriminator residual block, which is simpler than the generator's. Note that the last residual block does not apply downsampling.
###ℎ:= relu-conv-relu-downsample (𝑥)
###𝑥:= conv-downsample (𝑥)
 ###In the official BigGAN implementation, the architecture is slightly different for the first discriminator residual block, since it handles the raw image as input:
###ℎ:= conv-relu-downsample (𝑥)
###𝑥:= downsample-conv (𝑥)
###After these two steps, you can return the residual connection  ℎ+𝑥 .It is  notice that there is no class information in these residual blocks. As you'll see later in the code, the authors inject class-conditional information after the final hidden layer (and before the output layer) via channel-wise dot product.'''

## Discriminator
class DResidualBlock(nn.Module):
    '''
    DResidualBlock Class
    Values:
    in_channels: the number of channels in the input, a scalar
    out_channels: the number of channels in the output, a scalar
    downsample: whether to apply downsampling
    use_preactivation: whether to apply an activation function before the first convolution
    '''

    def __init__(self, in_channels, out_channels, downsample=True, use_preactivation=False):
        super().__init__()

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        self.activation = nn.ReLU()
        self.use_preactivation = use_preactivation  # apply preactivation in all except first dblock

        self.downsample = downsample    # downsample occurs in all except last dblock
        if downsample:
            self.downsample_fn = nn.AvgPool2d(2)
        self.mixin = (in_channels != out_channels) or downsample
        if self.mixin:
            self.conv_mixin = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))

    def _residual(self, x):
        if self.use_preactivation:
            if self.mixin:
                x = self.conv_mixin(x)
            if self.downsample:
                x = self.downsample_fn(x)
        else:
            if self.downsample:
                x = self.downsample_fn(x)
            if self.mixin:
                x = self.conv_mixin(x)
        return x

    def forward(self, x):
        # Apply preactivation if applicable
        if self.use_preactivation:
            h = F.relu(x)
        else:
            h = x

        h = self.conv1(h)
        h = self.activation(h)
        if self.downsample:
            h = self.downsample_fn(h)

        return h + self._residual(x)

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
    base_channels: the number of base channels, a scalar
    n_classes: the number of image classes, a scalar
    '''

    def __init__(self, base_channels=96, n_classes=1000):
        super().__init__()

        # For adding class-conditional evidence
        self.shared_emb = nn.utils.spectral_norm(nn.Embedding(n_classes, 16 * base_channels))

        self.d_blocks = nn.Sequential(
            DResidualBlock(3, base_channels, downsample=True, use_preactivation=False),
            AttentionBlock(base_channels),

            DResidualBlock(base_channels, 2 * base_channels, downsample=True, use_preactivation=True),
            AttentionBlock(2 * base_channels),

            DResidualBlock(2 * base_channels, 4 * base_channels, downsample=True, use_preactivation=True),
            AttentionBlock(4 * base_channels),

            DResidualBlock(4 * base_channels, 8 * base_channels, downsample=True, use_preactivation=True),
            AttentionBlock(8 * base_channels),

            DResidualBlock(8 * base_channels, 16 * base_channels, downsample=True, use_preactivation=True),
            AttentionBlock(16 * base_channels),

            DResidualBlock(16 * base_channels, 16 * base_channels, downsample=False, use_preactivation=True),
            AttentionBlock(16 * base_channels),

            nn.ReLU(inplace=True),
        )
        self.proj_o = nn.utils.spectral_norm(nn.Linear(16 * base_channels, 1))

    def forward(self, x, y=None):
        h = self.d_blocks(x)
        h = torch.sum(h, dim=[2, 3])

        # Class-unconditional output
        uncond_out = self.proj_o(h)
        if y is None:
            return uncond_out

        # Class-conditional output
        cond_out = torch.sum(self.shared_emb(y) * h, dim=1, keepdim=True)
        return uncond_out + cond_out

## Training
device = 'cpu'

# Initialize models
base_channels = 96
z_dim = 120
n_classes = 5   # 5 classes is used instead of the original 1000, for efficiency
shared_dim = 128
generator = Generator(base_channels=base_channels, bottom_width=4, z_dim=z_dim, shared_dim=shared_dim, n_classes=n_classes).to(device)
discriminator = Discriminator(base_channels=base_channels, n_classes=n_classes).to(device)

# Initialize weights orthogonally
for module in generator.modules():
    if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
        nn.init.orthogonal_(module.weight)
for module in discriminator.modules():
    if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
        nn.init.orthogonal_(module.weight)

# Initialize optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.999), eps=1e-6)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=4e-4, betas=(0.0, 0.999), eps=1e-6)

batch_size = n_classes

z = torch.randn(batch_size, z_dim, device=device)                 # Generate random noise (z)
y = torch.arange(start=0, end=n_classes, device=device).long()    # Generate a batch of labels (y), one for each class
y_emb = generator.shared_emb(y)                                   # Retrieve class embeddings (y_emb) from generator

x_gen = generator(z, y_emb)                                       # Generate fake images from z and y_emb
score = discriminator(x_gen, y)                                   # Generate classification for fake images


##BigGAN-deep
'''This modification of BigGAN is 4x deeper, sports a modified residual block architecture, and concatenates the entire vector to  𝑐 (as opposed to separate chunks at different resolutions).
Typically on a difficult and complex task that you're unlikely to overfit, you expect better performance when a model has more parameters, because it has more room to learn. Surprisingly, BigGAN-deep has fewer parameters than its BigGAN counterpart. Architectural optimizations such as using depthwise separable convolutions and truncating/concatenating channels in skip connections (as opposed to using pointwise convolutions) decrease parameters without trading expressivity.'''




