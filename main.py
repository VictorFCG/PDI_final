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
# Porcentagem da imagem que terá grãos
grain_amount = 0.0075
# Tamanho médio dos grãos  
grain_size = 0.5
# Intensidade dos grãos
grain_intensity = 15


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

def filmGrain(u, grain_amount, grain_size, grain_intensity):
    m, n, c = u.shape
    grain_image = np.zeros((m, n, c))

    # Cria grid de índices para as dimensões da imagem
    y_indices, x_indices = np.ogrid[:m, :n]
    
    num_grains = int(grain_amount * m * n)

    # Gera as posições e raios dos grãos aleatoriamente
    for _ in range(num_grains):
        # Seleciona aleatoriamente uma posição na imagem
        x = np.random.randint(0, m)
        y = np.random.randint(0, n)

        # Gera um raio aleatório para o grão usando dist. normal centrada em grain_size
        radius = np.random.normal(loc=grain_size, scale=grain_size / 2)

        # Cria uma máscara circular com centro em (x, y) com o raio calculado
        mask = (x_indices - y) ** 2 + (y_indices - x) ** 2 <= radius ** 2

        # Intensidade do grão aleatória dentro da faixa determinada por grain_intensity
        intensity = np.random.uniform(-grain_intensity, grain_intensity)
        
        # Aplica a intensidade em grain_image nas posições definidas pela máscara
        grain_image[mask] += intensity

    v = u + grain_image
    v = np.clip(v, 0, 1.0)

    return v

def main():
    input_img = cv2.imread("05.png").astype(np.float32) / 255

    yiq = colorRGB2YIQ(input_img)
    yiq = chromaDephase(yiq)
    yiq = blur(yiq)
    
    bgr = colorYIQRGB(yiq)    
    bgr = scanline(bgr)
    bgr = bloom(bgr)
    bgr = filmGrain(bgr, grain_amount, grain_size, grain_intensity)

    cv2.imshow("final", bgr)

    cv2.waitKey()
    cv2.imshow("convert", input_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================
