import os, cv2, numpy as np, torch, timm
CFG = {"tb": ("efficientnet_b0", ["normal", "tb_suspect"]),
       "dr": ("efficientnet_b3", ["no_dr", "mild", "moderate", "severe", "proliferative"]),
       "oral": ("mobilenetv3_large_100", ["normal", "opmd", "suspicious"])}
W = os.path.join(os.path.dirname(__file__), "..", "weights")

def load(d):
    arch, cls = CFG[d]; p = os.path.join(W, f"{d}.pt"); trained = os.path.exists(p)
    m = timm.create_model(arch, pretrained=not trained, num_classes=len(cls), drop_rate=0.3)
    if trained: m.load_state_dict(torch.load(p, map_location="cpu"))
    return m.eval(), trained

def prep(rgb, size=300):
    x = cv2.resize(rgb, (size, size)).astype("float32") / 255
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)[None]

@torch.no_grad()
def predict_mc(m, x, n=10):
    """MC-Dropout: keep only Dropout layers active; returns mean probs and std."""
    m.eval()
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout): mod.train()
    p = torch.stack([m(x).softmax(1)[0] for _ in range(n)]); m.eval()
    return p.mean(0).numpy(), p.std(0).numpy()

def gradcam(m, x, rgb, cls_idx):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    cam = GradCAM(model=m, target_layers=[m.conv_head])(
        input_tensor=x, targets=[ClassifierOutputTarget(cls_idx)])[0]
    base = cv2.resize(rgb, (cam.shape[1], cam.shape[0])).astype("float32") / 255
    return cam, show_cam_on_image(base, cam, use_rgb=True)
