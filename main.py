import numpy as np
import cv2
import math

# constantes de ajuste
#_________________________________________________________________________
#defasagem do chroma 0=não muda
bleeding_level = 1
#faixa de intensidade/aplicação do bloom
bloom_threshold = 0.5
#ganho de iluminação máximo
bloom_cap = 0.12
#scanline intensidade
sl_intensity=0.65

def bloom(input_img):
    base = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    scale = round(base.shape[1]/16)
    k = 1 + (2*scale)
    threshold = np.max(base) * bloom_threshold
    #brightpass
    bpass = np.where(base < threshold, 0, base)
    #blur
    total = bpass * 0
    for i in range(4):
        gblur = cv2.GaussianBlur(bpass, (k, k), 0)
        total += gblur    
    #limitando aumento de intensidade e testando parâmetros
    total *= bloom_cap
    total += 1
    for c in range(3):
        input_img[:, :, c] *= total    
    #truncamento
    input_img = np.clip(input_img, 0, 1.0)    
    return input_img

def blur(input_img):
    w = input_img.shape[1]
    scale = round(w/640)
    k = 1 + (2*scale)
    input_img = cv2.blur(input_img, (k, 1))
    
    return input_img    
    
def chromaDephase(input_img):
    if(bleeding_level != 0):
        w = input_img.shape[1]
        h = input_img.shape[0]
        scale = round(w/640)
        #chroma
        input_img[:, :, 1] = cv2.warpAffine(input_img[:, :, 1], np.float32([[1, 0, bleeding_level*scale], [0, 1, 0]]), (w, h))
        input_img[:, :, 2] = cv2.warpAffine(input_img[:, :, 2], np.float32([[1, 0, -bleeding_level*scale], [0, 1, 0]]), (w, h))
       
    return input_img

def ghosting(input_img):
    # ainda não deu certo

    return input_img


def colorYIQRGB(input_img):
    bgr = np.copy(input_img) * 0

    bgr[:, :, 0] = (
        (input_img[:, :, 0])
        - (1.106 * input_img[:, :, 1])
        + (1.703 * input_img[:, :, 2])
    )

    bgr[:, :, 1] = (
        (input_img[:, :, 0])
        - (0.272 * input_img[:, :, 1])
        - (0.647 * input_img[:, :, 2])
    )

    bgr[:, :, 2] = (
        (input_img[:, :, 0])
        + (0.956 * input_img[:, :, 1])
        + (0.619 * input_img[:, :, 2])
    )

    return bgr


def colorRGB2YIQ(input_img):
    yiq = np.copy(input_img) * 0

    yiq[:, :, 0] = (
        (0.299 * input_img[:, :, 2])
        + (0.587 * input_img[:, :, 1])
        + (0.144 * input_img[:, :, 0])
    )

    yiq[:, :, 1] = (
        (0.5959 * input_img[:, :, 2])
        - (0.2746 * input_img[:, :, 1])
        - (0.3213 * input_img[:, :, 0])
    )

    yiq[:, :, 2] = (
        (0.2115 * input_img[:, :, 2])
        - (0.5227 * input_img[:, :, 1])
        + (0.3112 * input_img[:, :, 0])
    )
    return yiq

def scanline(img):
    scanline_img = np.copy(img)
    w = img.shape[1]
    h = img.shape[0]
    scanline_img = cv2.resize(scanline_img, (w, 480))    
    # Itera de duas em duas linhas
    for i in range(0, scanline_img.shape[0], 2):
        # Diminui o brilho da linha
        scanline_img[i] = scanline_img[i] * sl_intensity
    
    
    scanline_img = cv2.resize(scanline_img, (w, h))
    return scanline_img

def filmGrain (img):
    #implementar
    return img

def main():
    input_img = cv2.imread("01.png").astype(np.float32) / 255

    yiq = colorRGB2YIQ(input_img)
    yiq = chromaDephase(yiq)
    yiq = blur(yiq)
    
    bgr = colorYIQRGB(yiq)    
    bgr = scanline(bgr)
    bgr = bloom(bgr)
    
    cv2.imshow("yiq", bgr)
    cv2.waitKey()
    cv2.imshow("convert", input_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================
