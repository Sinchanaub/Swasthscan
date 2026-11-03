"""python train.py tb data/tb  (folders: data/tb/{train,val}/<class>/*.jpg)"""
import sys, torch, torchvision as tv
from core.models import CFG, load
d, root = sys.argv[1], sys.argv[2]
T = lambda tr: tv.transforms.Compose(([tv.transforms.RandomHorizontalFlip(), tv.transforms.ColorJitter(.3, .3, .2),
    tv.transforms.RandomRotation(15)] if tr else []) + [tv.transforms.Resize((300, 300)), tv.transforms.ToTensor(),
    tv.transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
tr = torch.utils.data.DataLoader(tv.datasets.ImageFolder(f"{root}/train", T(1)), 32, shuffle=True)
va = torch.utils.data.DataLoader(tv.datasets.ImageFolder(f"{root}/val", T(0)), 32)
m, _ = load(d); opt = torch.optim.AdamW(m.parameters(), 3e-4); best = 0
for ep in range(10):
    m.train()
    for x, y in tr: opt.zero_grad(); torch.nn.functional.cross_entropy(m(x), y).backward(); opt.step()
    m.eval(); c = n = 0
    with torch.no_grad():
        for x, y in va: c += (m(x).argmax(1) == y).sum().item(); n += len(y)
    print(ep, c / n)
    if c / n > best: best = c / n; torch.save(m.state_dict(), f"weights/{d}.pt")
