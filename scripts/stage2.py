import argparse, os
from omegaconf import OmegaConf
from einops import rearrange
from torchvision import datasets
from torch import autocast
from contextlib import nullcontext
import sys
sys.path.append("./")
sys.path.append("../")
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from my_transforms import *
import warnings
warnings.filterwarnings("ignore")
dataset='cifar10' # STL10 cifar10

import torchvision.transforms as tfs

if dataset == "cifar10":
    transforms = tfs.Compose(
            [tfs.Resize(32), tfs.ToTensor()]
        )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../../Data/cifar10/', train=False,
                         download=True,
                         transform=transforms
                         ),
        batch_size=32,
        shuffle=True,
    )

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def from_img_generate_sample(sampler,model,c,z_enc,t_enc,scale,uc):
    with torch.no_grad():
        # decode it
        samples_ddim = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                 unconditional_conditioning=uc, )
        x_samples_ddim = model.decode_first_stage(samples_ddim)
    return x_samples_ddim

def saveImageto(x_samples_ddim, sample_path, class_id, epoch, i):
    x_samples_ddim = tfs.Resize(224)(x_samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    for x_sample in x_samples_ddim:  # x_checked_image_torch:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        img.save(os.path.join(sample_path, f"{class_id}/{epoch}_{i:04}.png"))

def train_transforms(inputs):
    img_size = inputs.size(-1)
    image_gap = random.randint(2, 8)
    random_trans = tfs.RandomOrder([
        tfs.RandomApply(
            [
                tfs.Resize((img_size + image_gap, img_size + image_gap)),
                tfs.CenterCrop((img_size, img_size)),
            ],0.5
        ),
        tfs.RandomHorizontalFlip(),
        tfs.RandomApply(
            [tfs.RandomRotation(image_gap)],0.5
        ),
        tfs.RandomApply(
            [HandV_translation(image_gap)],0.5
        ),
        tfs.RandomApply(
            [
                tfs.Pad([int(image_gap / 2), int(image_gap / 2),
                                  int(image_gap / 2), int(image_gap / 2)]),
                tfs.Resize((img_size, img_size)),
            ],0.5
        ),
        tfs.RandomApply(
            [tfs.GaussianBlur(3, sigma=(0.1, 1.0))],0.5
        ),
        tfs.RandomApply(
            [AddGaussianNoise(0.0, 1.0, 0.01)],0.5
        ),
        tfs.RandomApply(
            [AddSaltPepperNoise(0.01)],0.5
        ),
        tfs.RandomApply(
            [tfs.RandomAffine(image_gap)],0.5
        ),
        tfs.RandomApply(
            [tfs.RandomErasing(scale=(0.02, 0.22))],0.5
        ),
    ])
    return random_trans(inputs)

def mixup_data(x_s, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x_s.size()[0]
    if batch_size == 1:
        return x_s
    random_mix = random.randint(0, 1)
    if random_mix == 0:
        index0 = np.random.randint(0, batch_size-1)
        random_mix = random.randint(0, 1)
        if random_mix==0:
            a = train_transforms(x_s[index0:index0+1],[random.randint(0, 10)])
        else:
            a = train_transforms(x_s[index0:index0+1])
        mixed_x = a
    else:
        index0 = np.random.randint(0, batch_size-1)
        index1 = np.random.randint(0, batch_size-1)
        a = train_transforms(x_s[index0:index0+1])
        b = train_transforms(x_s[index1:index1+1])
        random_mix = random.randint(0, 1)
        if random_mix == 0:
            mixed_x = lam * a + (1 - lam) * b
        elif random_mix ==1:
            mixed_x, lam = CutMix(1.0)(a, b)
    return mixed_x

def main(opt):

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = os.path.join(opt.outdir)

    batch_size = opt.n_samples

    sample_path = os.path.join(outpath, "generate_images")
    os.makedirs(sample_path, exist_ok=True)

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    codebooks = torch.load(opt.codebookdir)
    print(f"load codebooks from '{opt.codebookdir}'")

    classnames = []
    with open(dataset+'.txt') as f:
        for clas in f.readlines():
            classnames.append(clas)
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                c1 = model.get_learned_conditioning(classnames)
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning([""]*batch_size)

    epoch = 0
    class_num = len(classnames)
    for i in range(class_num):
        os.makedirs(os.path.join(sample_path, str(i)),exist_ok=True)
    generate_num = 5 # 1 epoch generate image num

    while True:
        for class_id in range(class_num): # #
            text_cond=c1[class_id:class_id + 1].repeat(batch_size,1,1)
            for i in range(generate_num):
                init_latent = mixup_data(torch.cat([codebooks[class_id][li][1] for li in range(len(codebooks[class_id]))]))
                with torch.no_grad():
                    with precision_scope("cuda"):
                        with model.ema_scope():
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
                            x_samples_ddim = from_img_generate_sample(sampler, model, text_cond, z_enc, t_enc, opt.scale, uc)
                saveImageto(x_samples_ddim, sample_path, class_id, epoch, i)
            print(str(epoch) +" epoch, class=" + str(class_id) + ", completed")
        epoch+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-tmp"
    )
    parser.add_argument(
        "--codebookdir",
        type=str,
        nargs="?",
        help="dir of codebook",
        default="./codebooks/codebook_10_cifar10.pth"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="../models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    opt = parser.parse_args()

    main(opt)
