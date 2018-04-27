import torch
if __name__ == '__main__':
    torch.cuda.set_device(0)
    print torch.cuda.is_available()
    a = torch.randn(10).cuda();
    print a;
