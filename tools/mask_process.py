import cv2, numpy as np

mask = cv2.imread("test_result\X01_1_0000010900.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("mask",mask)
cv2.waitKey(0)

# 1.把 mask 变成「前方扇区安全度」
# 思路：把画面垂直切成 N 个扇区，统计每个扇区里 water 像素占比，得到 0-1 的分数列表。
N = 9                          # 奇数扇区，中间是正前方
h, w = mask.shape              # mask 0/1 单通道
sector_w = w // N
score = np.zeros(N, dtype=np.float32)
for i in range(N):
    roi = mask[:, i*sector_w:(i+1)*sector_w]
    score[i] = roi.mean()      # water 占比
print("score: ", score)


# 2.选最安全扇区 → 舵角
# 取分数最高的扇区编号 best = np.argmax(score)，映射到舵角：
best = np.argmax(score)
angle_per_sector = 60 / N                       # 每扇区角度
steer_angle = (best - N // 2) * angle_per_sector  # 中心为 0°
# 限幅 ±30°
steer_angle = np.clip(steer_angle, -30, 30)
print("当前帧计算出的舵角：", steer_angle)


# 3. 生成速度
# 用「最远可行距离」当 PID 设定值：
# 把 mask 下半部分做行累加，最大连续 water 行数 → 速度 PWM。
# 远快近慢，防止撞岸。


