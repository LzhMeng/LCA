import argparse, os
import torch
from torch import nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from torchvision import datasets
from torch import autocast
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import sys
sys.path.append("./")
sys.path.append("../")
import warnings
warnings.filterwarnings("ignore")
dataset = "cifar10"# STL10 cifar10

import torchvision.transforms as tfs
import torch.nn.functional as F

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

def add_noise(x,stddev=0.025):
    noise = stddev * torch.randn(x.shape).cuda()
    x_noisy = torch.clamp(x + noise, 0.0, 1.0)
    return x_noisy

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

def from_text_generate_sample(sampler,model,c,scale,uc):
    with torch.no_grad():
        samples_ddim, _ = sampler.sample(S=50,
                                         conditioning=c,
                                         batch_size=1,
                                         shape=[4,64,64],
                                         verbose=False,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=uc,
                                         eta=0.0)
        x_samples_ddim = model.decode_first_stage(samples_ddim)
    return x_samples_ddim

def saveImageto(x_samples_ddim, sample_path, class_id, epoch, i):
    os.makedirs(os.path.join(sample_path, str(class_id)), exist_ok=True)
    x_samples_ddim = tfs.Resize(224)(x_samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    for x_sample in x_samples_ddim:  # x_checked_image_torch:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        img.save(os.path.join(sample_path, f"{class_id}/{epoch}_{i:04}.png"))


def main(opt):

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = os.path.join(opt.outdir)

    sample_path0 = os.path.join(outpath, "all_samples")
    sample_path1 = os.path.join(outpath, "member_samples")
    os.makedirs(sample_path0, exist_ok=True)
    os.makedirs(sample_path1, exist_ok=True)

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)

    classnames = []
    with open(dataset+'.txt') as f:
        for clas in f.readlines():
            clas = clas.replace('\n','')
            clas = clas.replace('_',' ')
            classnames.append(clas)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                uc = None
                c0 = model.get_learned_conditioning(classnames)
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning([""])

    epoch = 0
    class_num = len(classnames)

    generate_num = 5 # 1 epoch generate image num
    save_num = 10 # length of codebook
    codebooks = [[] for _ in classnames] # init

    from models.resnet import resnet34
    original_net = resnet34(10).cuda()
    state_dict = torch.load(opt.classify_model)
    original_net.load_state_dict(state_dict)
    original_net = nn.DataParallel(original_net)
    original_net.eval()
    isallright=0
    AllQuery=0
    while True:
        for class_id in range(class_num): # #
            save_class_path = os.path.join(sample_path1, str(class_id))
            os.makedirs(save_class_path,exist_ok=True)
            class_codebook = codebooks[class_id]
            class_codebook = sorted(class_codebook,key=lambda x:x[0])  # sort codebook
            if len(class_codebook) >= save_num:
                if class_codebook[-1][0] < 1e-3:
                    continue
                else:
                    isallright += 1

            text_cond=c0[class_id:class_id + 1]

            base_count = len(os.listdir(save_class_path))
            for i in range(generate_num):
                with torch.no_grad():
                    with precision_scope("cuda"):
                        with model.ema_scope():
                            x_samples_ddim = from_text_generate_sample(sampler, model, text_cond, opt.scale, uc)
                saveImageto(x_samples_ddim, sample_path0, class_id, epoch, i)
                img0 = Image.open(os.path.join(sample_path0, f"{class_id}/{epoch}_{i:04}.png"))
                img = transforms(img0).unsqueeze(0).cuda()
                with torch.no_grad():
                    outputs1 = original_net(img)
                    outputs1 = F.softmax(outputs1, dim=1)
                    prob, label1 = torch.max(outputs1.data, 1)
                    AllQuery+=1
                if label1 == class_id and prob>0.8:
                    with torch.no_grad():
                        adv_img = add_noise(img, 0.05)
                        outputs2 = original_net(adv_img)
                        outputs2 = F.softmax(outputs2, dim=1)
                        _, label2 = torch.max(outputs2.data, 1)
                        MIA = nn.MSELoss()(outputs1, outputs2)
                        AllQuery+=1
                    if label1 == label2:
                        init_image = x_samples_ddim.cuda()
                        encode_1 = model.encode_first_stage(init_image)
                        init_latent = model.get_first_stage_encoding(encode_1)  # move to latent space
                        if len(class_codebook) < save_num:  # delete out data(bad)
                            img0.save(os.path.join(save_class_path, f"{epoch}_{base_count:04}.png"))
                            class_codebook.append([MIA.item(), init_latent, f"{epoch}_{base_count:04}.png"])
                        else:
                            img0.save(os.path.join(save_class_path, f"{epoch}_{base_count:04}.png"))
                            class_codebook.append([MIA.item(), init_latent, f"{epoch}_{base_count:04}.png"])
                            class_codebook = sorted(class_codebook,key=lambda x:x[0])  # sort codebook
                            os.remove(os.path.join(save_class_path, class_codebook[-1][-1]))
                            del class_codebook[-1]
                        base_count += 1

                if len(class_codebook) > save_num:
                    class_codebook = sorted(class_codebook,key=lambda x:x[0])  # ,key=lambda x:x[1])  # sort codebook
                    os.remove(os.path.join(save_class_path, class_codebook[-1][-1]))
                    del class_codebook[-1]

            codebooks[class_id] = class_codebook
            print(str(epoch) +" epoch, class=" + str(class_id) +
                  ", completed, now have "+str(len(class_codebook))+" good sample.")
            torch.save(codebooks, f"./codebooks/codebooks_{dataset}.pth")

        if isallright==0:
            print("All Right! ")
            break
        else:
            print("Query num: "+str(AllQuery))
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
        help="dir of codebook to save",
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
        "--target_model",
        type=str,
        default="../pretrained/resnet34_cifar10.pth",
        help="path to checkpoint of target model",
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
