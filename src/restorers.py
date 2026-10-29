# Track 1 per-image restorers (UNet and NAFNet variants), inference only.
# Copied verbatim from the training modules t1_restnet / t1_restnet2 / t1_rest_ens / t1_rest_ab that produced the
# submission, with their module constants fixed to the values the submission build ran with:
#   GSTATS=1 (12 global colour statistics per image), PHYS=1 in the input builder (t1_rest_ens sets it), LEVELS=4,
#   NAF: width 32, encoder blocks 2,2,4,8, 12 middle blocks, decoder blocks 2,2,2,2; UNet width 64; no unsharp mask.
# Every input channel is computed from the image itself and its own file-name key (water type, depth).
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from macvi.eurs import rsuigm

GSTATS = True
CIN = 17 + 12 * GSTATS          # cubic map 3 + degraded 3 + water one-hot 10 + depth 1 + global stats 12 = 29
LEVELS = 4
NW, NENC, NMID, NDEC = 32, (2, 2, 4, 8), 12, (2, 2, 2, 2)
UNET_WIDTH = 64


def gstats(deg):
    x = deg.reshape(-1, 3).astype(np.float32) / 255
    return np.concatenate([x.mean(0), x.std(0), np.percentile(x, 1, 0), np.percentile(x, 99, 0)])


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


def compose(x, o, phys):
    """Output from the head: residual on the cubic map, or (physics head) gated inversion + residual."""
    if not phys:
        return (x[:, :3] + o).clamp(0, 1)
    x = x.float()  # channels: cubic 0-2, degraded 3-5, cond 6-28, e 29-31, Bz 32-34, beta 35
    t = torch.exp(-x[:, 35:36] * F.softplus(o[:, 3:4] - 0.7))  # d starts near its train mean (~0.4)
    j0 = ((x[:, 3:6] - x[:, 32:35] * (1 - t)) / (x[:, 29:32] * t).clamp_min(0.02)).clamp(0, 1)
    g = torch.sigmoid(o[:, 4:5])
    return (g * j0 + (1 - g) * x[:, :3] + o[:, :3]).clamp(0, 1)


class UNet(nn.Module):
    """t1_restnet's UNet with LEVELS levels (channels w, 2w, 4w, 8w, then capped at 8w); LEVELS=4 is the same net."""
    def __init__(self, cin, w=32, levels=4, phys=False):
        super().__init__()
        self.phys = phys
        ch = [w * min(2 ** i, 8) for i in range(levels)]
        self.enc = nn.ModuleList([block(cin, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(levels - 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2) for i in range(levels - 1)])
        self.dec = nn.ModuleList([block(2 * ch[i], ch[i]) for i in range(levels - 1)])
        self.head = nn.Conv2d(ch[0], 3 + 2 * phys, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skips, h = [], x
        for i, e in enumerate(self.enc):
            h = e(h if i == 0 else F.max_pool2d(h, 2))
            skips.append(h)
        for i in reversed(range(len(self.dec))):
            h = self.dec[i](torch.cat([self.up[i](h), skips[i]], 1))
        return compose(x, self.head(h).float(), self.phys)


class LN2d(nn.Module):
    """LayerNorm over channels, statistics in float32."""
    def __init__(self, c):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c))

    def forward(self, x):
        u = x.float()
        mu = u.mean(1, keepdim=True)
        y = ((u - mu) * torch.rsqrt((u - mu).pow(2).mean(1, keepdim=True) + 1e-6)).to(x.dtype)
        return y * self.w.view(1, -1, 1, 1).to(x.dtype) + self.b.view(1, -1, 1, 1).to(x.dtype)


class NAFBlock(nn.Module):
    """NAFNet block: LN, 1x1 expand, 3x3 depthwise, SimpleGate, channel attention, 1x1; then a SimpleGate FFN."""
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = LN2d(c), LN2d(c)
        self.c1, self.c2 = nn.Conv2d(c, 2 * c, 1), nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sca, self.c3 = nn.Conv2d(c, c, 1), nn.Conv2d(c, c, 1)
        self.c4, self.c5 = nn.Conv2d(c, 2 * c, 1), nn.Conv2d(c, c, 1)
        self.beta, self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1)), nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        a, b = self.c2(self.c1(self.n1(x))).chunk(2, 1)
        y = a * b
        y = self.c3(y * self.sca(F.adaptive_avg_pool2d(y, 1)))
        x = x + y * self.beta.to(y.dtype)
        a, b = self.c4(self.n2(x)).chunk(2, 1)
        return x + self.c5(a * b) * self.gamma.to(x.dtype)


class NAF(nn.Module):
    """NAFNet-style UNet over the same input channels; 3 (+2 physics) output channels through compose()."""
    def __init__(self, cin, w=32, enc=NENC, mid=12, dec=NDEC, phys=False):
        super().__init__()
        self.phys = phys
        self.intro = nn.Conv2d(cin, w, 3, padding=1)
        self.encs, self.downs, self.ups, self.decs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = w
        for n in enc:
            self.encs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, 2)); c *= 2
        self.mid = nn.Sequential(*[NAFBlock(c) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2))); c //= 2
            self.decs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.head = nn.Conv2d(w, 3 + 2 * phys, 3, padding=1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h, skips = self.intro(x), []
        for e, d in zip(self.encs, self.downs):
            h = e(h); skips.append(h); h = d(h)
        h = self.mid(h)
        for u, dc, s in zip(self.ups, self.decs, reversed(skips)):
            h = dc(u(h) + s)
        return compose(x, self.head(h).float(), self.phys)


def make_net(arch, phys):
    cin = CIN + 7 * phys
    if arch == "naf":
        return NAF(cin, NW, NENC, NMID, NDEC, phys)
    return UNet(cin, UNET_WIDTH, LEVELS, phys)


class GpuInputs:
    """Input channels on the GPU: cubic map (per-key 13x3 matrix), degraded, water one-hot, depth/100, global stats of the
    whole image, and the RSUIGM key tables (illumination e_c, veiling light Bz_c, beta_w) at the image's own key."""
    def __init__(self, maps, params_path):
        self.keys = sorted(maps)
        self.kid = {k: i for i, k in enumerate(self.keys)}
        self.A = torch.tensor(np.stack([maps[k] for k in self.keys]), dtype=torch.float32, device="cuda")
        p = rsuigm.load_params(params_path)
        tab = lambda name, k: rsuigm._interp_z(p[name][int(np.clip(k[0], 1, 10)) - 1], p["zs"], k[1])  # noqa: E731
        self.phys = torch.tensor(np.stack([np.concatenate([tab("e", k), tab("Bz", k) / 255.0,
                                                           [p["beta"][int(np.clip(k[0], 1, 10)) - 1]]])
                                           for k in self.keys]), dtype=torch.float32, device="cuda")  # (K, 7)

    def __call__(self, d, keys, stats):
        """d: (N, 3, H, W) float 0-1; keys: list of (water, depth); stats: (N, 12) tensor."""
        r, g, b = d[:, 0:1], d[:, 1:2], d[:, 2:3]
        feats = torch.cat([d, torch.ones_like(r), d * d, r * g, r * b, g * b, d * d * d], 1)  # = t1_rest_ab.design "cub"
        A = self.A[torch.tensor([self.kid[k] for k in keys], device="cuda")]  # (N, 13, 3)
        cub = torch.einsum("nfhw,nfc->nchw", feats, A).clamp(0, 1)
        n, _, h, w = d.shape
        cond = torch.zeros(n, 23, h, w, device="cuda")
        for j, (wt, dp) in enumerate(keys):
            cond[j, min(max(int(wt), 1), 10) - 1] = 1
            cond[j, 10] = dp / 100.0
        cond[:, 11:] = stats[:, :, None, None]
        ph = self.phys[torch.tensor([self.kid[k] for k in keys], device="cuda")][:, :, None, None].expand(-1, -1, h, w)
        return torch.cat([cub, d, cond, ph], 1)


def load(path, arch, phys):
    net = make_net(arch, phys).cuda().eval()
    net.load_state_dict(torch.load(path, map_location="cuda"))
    return net, phys


@torch.no_grad()
def run(net, phys, x):
    xi = x if phys else x[:, :CIN]
    with torch.autocast("cuda", torch.float16):
        return net(xi).float()


def restore_mean(nets, gin, deg, k):
    """Mean of the restorers on one degraded uint8 HxWx3 image with key k; float HxWx3 in 0-1."""
    h, w = deg.shape[:2]
    d = torch.from_numpy(deg).cuda().permute(2, 0, 1)[None].float() / 255
    d = F.pad(d, (0, (-w) % 16, 0, (-h) % 16), mode="reflect")
    x = gin(d, [k], torch.from_numpy(gstats(deg).astype(np.float32))[None].cuda())
    y = torch.stack([run(n, ph, x)[0, :, :h, :w] for n, ph in nets]).mean(0).clamp(0, 1)
    return y.permute(1, 2, 0).cpu().numpy()
