import pygame, sys
import numpy as np
from keras.models import load_model
import cv2

BOUNDRYINC = 5
WINDOWSIZEX = 640
WINDOWSIZEY = 480

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False

MODEL = load_model("mnist.keras")

LABELS = {0: "Zero", 1: "One", 2: "Two",
          3: "Three", 4: "Four", 5: "Five",
          6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

pygame.init()

DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Board")

iswriting = False
number_xcord = []
number_ycord = []

PREDICT = True

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)
        if event.type == pygame.MOUSEBUTTONDOWN:
            iswriting = True
        if event.type == pygame.MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)
            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDRYINC)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDRYINC, 0), min(WINDOWSIZEY, number_ycord[-1] + BOUNDRYINC)

            # Collect pixel data from the rectangle region
            img_arr = pygame.surfarray.array3d(DISPLAYSURF)
            img_crop = img_arr[rect_min_x:rect_max_x, rect_min_y:rect_max_y, :].astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png", img_crop)
                image_cnt += 1

            if PREDICT:
                # Resize and preprocess the image
                image = cv2.cvtColor(cv2.resize(img_crop, (28, 28)), cv2.COLOR_BGR2GRAY)
                image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
                image = image / 255.0

                # Ensure correct shape and predict using the loaded model
                image = image.reshape(1, 28, 28, 1)
                label = LABELS[np.argmax(MODEL.predict(image))]

                # Render text on the display surface
                font = pygame.font.Font(None, 36)
                text_surface = font.render(label, True, RED)
                text_rect = text_surface.get_rect()
                text_rect.center = (WINDOWSIZEX // 2, WINDOWSIZEY - 20)
                DISPLAYSURF.blit(text_surface, text_rect)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()
