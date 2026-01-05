import cv2, torch, time, os
from PIL import Image
from utils.visualize import get_color_pallete   # 作者自带的调色函数
from models.fast_scnn import get_fast_scnn      # 作者目录结构
import numpy as np


CAMERA_IDX = 3 # 0 为笔记本自带摄像头


# ---- 1. 加载模型（只跑一次） ----
# device = torch.device('cuda')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_fast_scnn('citys', pretrained=True,
                      root='./weights', map_location='cpu').to(device)
model.eval()

# ---- 2. 打开相机 ----
cap = cv2.VideoCapture(CAMERA_IDX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1024)          # 可按需要改分辨率
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
std  = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

# ---- 3. 主循环 ----
fps = 0
with torch.no_grad():
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        # BGR→RGB & 转 PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # 同作者 demo 的预处理：resize 768×1536 → ToTensor → normalize
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((768, 1536)),   # 和训练保持一致
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        tensor = transform(pil_img).unsqueeze(0).to(device)

        # 推理
        out = model(tensor)[0]          # out shape: [19, H, W]
        pred = out.argmax(1).squeeze(0).cpu().numpy()  # [H, W] 0~18

        show_color_mask = True # 显示灰度还是彩色结果
        if not show_color_mask:
            # 上色
            color_mask = get_color_pallete(pred, 'citys')   # PIL Image
            color_mask = cv2.cvtColor(np.array(color_mask), cv2.COLOR_RGB2BGR)

            # 原图与掩码同屏显示（可自由叠加/融合）
            color_mask = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]))
        else:
            # 1. 先拿到调色板图像（P 模式）
            color_pal = get_color_pallete(pred, 'citys')   # 仍是 P 模式
            # 2. 转 RGB 三通道
            color_pal = color_pal.convert('RGB')           # 现在每个像素是 (R,G,B)
            # 3. 再转 numpy + BGR
            color_mask = cv2.cvtColor(np.array(color_pal), cv2.COLOR_RGB2BGR)
            # 4. resize 到原图大小
            color_mask = cv2.resize(color_mask,
                                    (frame.shape[1], frame.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

        # 计算 FPS
        fps = 0.9 * fps + 0.1 * (1 / (time.time() - t0))
        cv2.putText(frame, f'FPS:{fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('original', frame)
        cv2.imshow('fast-scnn', color_mask)

        if cv2.waitKey(1) & 0xFF == 27:    # ESC 退出
            break

cap.release()
cv2.destroyAllWindows()