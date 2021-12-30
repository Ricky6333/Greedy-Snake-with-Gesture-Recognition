import random, os, math, torch
import pygame
import cv2, dlib
import numpy as np
import pandas as pd
from torch import nn
from torchvision import transforms
from PIL import Image


# 神經網路
class NeuronNetwork(nn.Module):
    def __init__(self):
        super().__init__()                      # 父類別初始函數
        self.flatten = nn.Flatten()             

        self.layer1 = nn.Sequential(            # input (,1,130,130)
            nn.Conv2d( in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0 ),  # out (64, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # out (,64, 64, 64)
        )

        self.layer2 = nn.Sequential(            
            nn.Conv2d( in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0 ), # out (,32, 62, 62)  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # out (,32, 31, 31)
        )


        self.fc1 = nn.Linear( in_features=32*31*31, out_features=128 )
        self.fc2 = nn.Linear(128, 4)
        

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = nn.ReLU()(out)
        out = nn.Dropout(0.5)(out)
        out = self.fc2(out)
        out = nn.Softmax(dim=1)(out)
        return out

# "D:/New_Python_Learning/Snake_Project/Game_test/model/model.pth"
model = torch.load( "./model/model.pth" )
classes = [ "up", "down", "right", "left"]


# 設定鏡頭-----------------------------------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    # 使用攝影機(cv2.CAP_DSHOW 可不顯示錯誤訊息)
# 設定顯示影像視窗
cv2.namedWindow("frame1",0)
cv2.resizeWindow('frame1', 500, 500)
cv2.namedWindow("frame2",0)
cv2.resizeWindow('frame2', 500, 500)

# 裁切區域的 x 與 y 座標（左上角）
img_x = 100
img_y = 100
# 裁切區域的長度與寬度
img_w = 230
img_h = 230

# 圖片轉換設定----------------------------------------------------------------------------
# 重新scale 圖片
Resize = transforms.Resize( [ 130, 130 ] )
ToTensor = transforms.ToTensor() # 圖片轉Tensor
Normalize = transforms.Normalize((0.5,), (0.5,)) # 圖片標準化
Gray_scale = transforms.Grayscale()

def Rescale_Img(img):
    # print('rescale')
    # img = Resize(img)      # 重新scale 圖片
    # print('rescale ok')
    # img = Gray_scale(img)
    img = ToTensor(img)    # 圖片轉Tensor
    # print('tensor ok')
    img = Normalize(img)   # 圖片標準化
    img = img.unsqueeze(0) # 圖片增加維度
    return img


#------------------------------------------------------------------------------------
pygame.init()
# 視窗設定
screen_width = 700
screen_height = 600
screen = pygame.display.set_mode((screen_width,screen_height))
clock = pygame.time.Clock() # 時間控制


# 變數設定
FPS = 60

# 顏色
RED = (255,0,0)
GREEN = (0,255,0)
BLACK = (0,0,0)
WHITE = (255,255,255)
YELLOW = (255,255,87)
GREY = (150,150,150)


# 蛇設定 (頭、身體還會在主畫面設定初始化)
snake_size = 15
snake_speed = 5
snake_body = [(100,100),(80,100),(60,100),(40,100)]   # 蛇身體
snake_head = snake_body[0]                  # 蛇頭
snake_len = len(snake_body)    


# 食物物件
class Food():
    def __init__(self,x,y):
        self.img = pygame.Surface((20,20))
        self.img.fill(GREEN)
        self.rect = self.img.get_rect()
        self.rect.x = x
        self.rect.y = y

    def update(self):
        screen.blit(self.img, self.rect)

# 顯示文字函數 and 設定
def text_draw(text, font, color, x,y):
    img = font.render(text, True, color)
    screen.blit(img, (x,y))
title_font = pygame.font.SysFont('Bauhaus 93', 70)  # title 字形
title = 'Greedy Snake'

# 遊戲迴圈
food_check = False  # 確認是否有食物
predict_list = []   # 預測列表
pred_counts = 0     # 預測解果數目
menu = 0
direction = 'RIGHT' # 預設往右移動
predict = ''        # 預測值
running = True
while(running):
    clock.tick(FPS)
    screen.fill(WHITE)

    # 取得事件------------------------------------------------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            """if (event.key == pygame.K_SPACE) and (menu != 1):
                menu += 1"""
            if (event.key == pygame.K_SPACE) :
                if menu == 1:
                    menu = 0
                elif menu == 0:
                    menu = 1

    # 顯示主菜單=================================
    if menu == 0:
        text_draw(title,title_font,BLACK, 140,screen_height/4)           # 印出標題

        # 初始化遊戲
        snake_body = [(80,100),(60,100),(40,100)]   # 蛇身體
        snake_head = snake_body[0]                  # 蛇頭
        direction = 'RIGHT'
        predict = ''
        food_check = False

    elif menu == 1:
        # 生成食物
        if food_check==False:
            x = random.randrange(0,screen_width-20)
            y = random.randrange(0,screen_height-20)
            food = Food(x,y)
            food_check = True


        # 取得預測方向
        if pred_counts == 7:
            predict = max(predict_list, key=predict_list.count)
            predict_list = []
            pred_counts = 0


        # 蛇移動和自我碰撞-----------------------------------------------------------------------
        # 取得蛇移動方向
        # key_pressed = pygame.key.get_pressed()
        if (predict == "right") and (direction !='LEFT'):    # 往右
            direction = 'RIGHT'  
        if (predict == "left") and (direction !='RIGHT'):    # 往左
            direction = 'LEFT'
        if (predict == "up") and (direction !='DOWN'):    # 往上
            direction = 'UP'
        if (predict == "down") and (direction !='UP'):    # 往下
            direction = 'DOWN'

        # 蛇移動 (蛇頭位置更新)
        if direction == 'RIGHT':
            snake_head = (snake_head[0]+snake_speed, snake_head[1])
        elif direction == 'LEFT':
            snake_head = (snake_head[0]-snake_speed, snake_head[1])
        elif direction == 'UP':
            snake_head = (snake_head[0], snake_head[1]-snake_speed)
        elif direction == 'DOWN':
            snake_head = (snake_head[0], snake_head[1]+snake_speed)

        # 加蛇頭、去蛇尾
        snake_body.insert(0,snake_head)
        snake_body.pop(len(snake_body)-1)
        head_rect = pygame.Rect(snake_head[0],snake_head[1],snake_size,snake_size)  # 蛇頭 Rect設定

        # 與食物碰撞
        if pygame.Rect.colliderect(head_rect, food.rect):
            if direction == 'RIGHT':
                snake_head = (snake_head[0]+10, snake_head[1])
            if direction == 'LEFT':
                snake_head = (snake_head[0]-10, snake_head[1])
            if direction == 'UP':
                snake_head = (snake_head[0], snake_head[1]-10)
            if direction == 'DOWN':
                snake_head = (snake_head[0], snake_head[1]+10)
            snake_body.insert(0,snake_head)                     # 增加蛇長度
            del food                                            # 刪除食物
            food_check = False


        # 邊界碰撞
        if snake_head[0]+(snake_size) >= screen_width or snake_head[0]<=0:
            menu = 0
        if snake_head[1]+(snake_size) >= screen_height or snake_head[1]<=0:
            menu = 0


        # 顯示更新-------------------------------------------------------------------------------
        if food_check==True:                    # 如果食物存在
            food.update()                       # 顯示食物
        snake_len = len(snake_body)             # 蛇長度更新
        for i in range(snake_len):    # 顯示蛇
                snake_Rect = pygame.Rect(snake_body[i][0],snake_body[i][1],snake_size, snake_size)
                pygame.draw.rect(screen, BLACK, snake_Rect)



    # 螢幕更新-------------------------------------------------------------------------------
    pygame.display.update() 

    # 辨識鏡頭設定---------------------------------------------------------------------------
    ret, img = cap.read()   # 讀取鏡頭圖片
    img = cv2.flip(img,1)                                                 # 翻轉鏡頭
    img = img[img_y:img_y+img_h, img_x:img_x+img_w]                       # 裁切圖片
    img_gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)                     # 轉為灰階
    img_blur = cv2.GaussianBlur( img_gray, (3, 3), 0)                     # 模糊化
    ret, img_th = cv2.threshold( img_blur, 105, 255, cv2.THRESH_BINARY)   # 二值化    
    
    cv2.imshow("frame1", img)    # 顯示鏡頭圖片(即時影像)
    cv2.imshow('frame2', img_th) # 顯示二值化圖片


    # 圖片辨識-------------------------------------------------------------------------------
    img_th = cv2.resize(img_th, (130, 130), interpolation=cv2.INTER_AREA) # 縮小
    img_tensor = Rescale_Img(img_th) 
    img_tensor = img_tensor.to("cuda")
    pred = model(img_tensor)

    max_num = torch.max(pred[0])
    idx = (pred[0] == max_num).nonzero()

    # predict = classes[idx]
    if menu == 1:
        predict_list.append(classes[idx])
        pred_counts += 1





# 結束
cap.release()           # 釋放(關閉)鏡頭
cv2.destroyAllWindows() # 關閉 Opencv 視窗
pygame.quit()   # 結束遊戲(關閉視窗)