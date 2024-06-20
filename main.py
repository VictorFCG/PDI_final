import numpy as np
import cv2
import math
import time

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
# Porcentagem da imagem que terá grãos
grain_amount = 0.3
# Intensidade dos grãos
grain_intensity = 0.15
# Fator de curvatura
strength = 0.1


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


def generateGaussianNoiseMask(shape, grain_intensity):
    m, n, c = shape
    # Gera ruído Gaussiano
    noise_mask = np.random.normal(loc=0, scale=grain_intensity, size=(m, n, c))
    return noise_mask


def applyNoiseMask(img, noise_mask, grain_amount):
    # Aplica a máscara de ruído à imagem
    noisy_img = img + grain_amount * noise_mask
    noisy_img = np.clip(noisy_img, 0, 1.0)
    return noisy_img


def applyCurvedBorderEffect(img, strength):
    height, width = img.shape[:2]
    # Gera grid com centro em (0, 0) para aplicar simetricamente a distorção de "barril"
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))

    # Calcula o raio (distância do centro)
    r = np.sqrt(x**2 + y**2)

    # Calcula o ângulo da coordenada polar
    theta = np.arctan2(y, x)
    # Aplica a distorção de barril
    rad = r + strength * r**2
    
    # Normaliza as coordenadas 
    map_x = width * (rad * np.cos(theta) + 1) / 2
    map_y = height * (rad * np.sin(theta) + 1) / 2
    
    # Remapeia a imagem usando as coordenadas
    distorted_image = cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    return distorted_image


def main():
    input_img = cv2.imread("05.png").astype(np.float32) / 255

    start_time = time.process_time()

    yiq = colorRGB2YIQ(input_img)
    yiq = chromaDephase(yiq)
    yiq = blur(yiq)
    
    bgr = colorYIQRGB(yiq)    
    bgr = scanline(bgr)
    bgr = bloom(bgr)

    noise_mask = generateGaussianNoiseMask(bgr.shape, grain_intensity)
    bgr = applyNoiseMask(bgr, noise_mask, grain_amount)
    bgr = applyCurvedBorderEffect(bgr, strength)

    cv2.imshow("final", bgr)

    print(f"--- {time.process_time() - start_time:.6f} seconds ---")

    cv2.waitKey()
    cv2.imshow("convert", input_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================
