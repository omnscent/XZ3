import torchvision
from torch.utils import data
from torchvision import transforms

def get_dataloader_workers():
    return 0


def load_BSDS300_data(batch_size=1):
    LR_trans = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(),transforms.Resize([80, 120])])
    HR_trans = transforms.Compose([transforms.ToTensor(),transforms.Grayscale()])
    LR_img = torchvision.datasets.ImageFolder(
        "./data/BSDS300/images", transform=LR_trans
    )
    HR_img = torchvision.datasets.ImageFolder(
        "./data/BSDS300/images", transform=HR_trans
    )
    LR_dataloader = data.DataLoader(
        LR_img, batch_size, num_workers=get_dataloader_workers()
    )
    HR_dataloader = data.DataLoader(
        HR_img, batch_size, num_workers=get_dataloader_workers()
    )
    return LR_dataloader, HR_dataloader


def load_SET11_modified_data(batch_size=1,resize=None):
    LR_trans = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(),transforms.Resize(128)])
    HR_trans = [transforms.ToTensor(),transforms.Grayscale()]
    if resize:
        HR_trans.append(transforms.Resize(resize))
    HR_trans = transforms.Compose(HR_trans)
    LR_img = torchvision.datasets.ImageFolder("./data/SET11", transform=LR_trans)
    HR_img = torchvision.datasets.ImageFolder("./data/SET11", transform=HR_trans)
    LR_dataloader = data.DataLoader(
        LR_img, batch_size, num_workers=get_dataloader_workers()
    )
    HR_dataloader = data.DataLoader(
        HR_img, batch_size, num_workers=get_dataloader_workers()
    )
    return LR_dataloader, HR_dataloader
