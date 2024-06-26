import numpy as np
import cv2
import math
import time

# constantes de ajuste
#_________________________________________________________________________
#defasagem do chroma 0=não muda
bleeding_level = 2
#faixa de intensidade/aplicação do bloom
bloom_threshold = 0.5
#ganho de iluminação máximo
bloom_cap = 0.12
#scanline intensidade
sl_intensity = 0.5
# Porcentagem da imagem que terá grãos
grain_amount = 0.3
# Intensidade dos grãos
grain_intensity = 0.15
# Fator de curvatura
curve_strength = 0.1
# Fator de escurescimento
darkening_strength = 0.2
#Rainbow effect - multiplicador de freq da função periódica
rb_mult = 18
#Rainbow effect - intensidade do efeito - 0=off
rb_int = 0.3
#bright comp
bcomp = 1.3
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


def scanline(img, field):
    scanline_img = np.copy(img)
    w = img.shape[1]
    h = img.shape[0]
    scale = round(h/480)
    print(scale)
    # Itera de duas em duas linhas
    for i in range(field*scale, scanline_img.shape[0] - scale, 2*scale):
        # Diminui o brilho da linha
        for j in range(scale):
            scanline_img[i+j] *= sl_intensity   
    
    scanline_img = cv2.resize(scanline_img, (w, h))
    return scanline_img


def gaussianNoiseMask(shape, grain_intensity):
    m, n, c = shape
    # Gera ruído Gaussiano
    noise_mask = np.random.normal(loc=0, scale=grain_intensity, size=(m, n, c))
    return noise_mask


def applyNoiseMask(img, noise_mask, grain_amount):
    # Aplica a máscara de ruído à imagem
    noisy_img = img + grain_amount * noise_mask
    noisy_img = np.clip(noisy_img, 0, 1.0)
    return noisy_img


def coordinatesGrid(width, height):
    # Gera grid com centro em (0, 0) das coordenadas
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    # Calcula o raio (distância do centro) dos pixels
    distance = np.sqrt(x**2 + y**2)     
    return x, y, distance


def curvedBorders(img, strength):
    height, width = img.shape[:2]

    x, y, r = coordinatesGrid(width, height)

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

    
def darkerBorders(img, strength):
    height, width = img.shape[:2]

    _, _, distance = coordinatesGrid(width, height)
    
    # Cria a máscara com base na distância
    mask = np.clip(1 - strength * distance, 0, 1)
    
    # Aplica a máscara
    mask = np.stack((mask,) * 3, axis=-1)
    img *= mask
    
    return img

def rainbowEffect(input_img):
    # Aplicar uma função periódica em um canal de cor
    if rb_int > 0:
        w = input_img.shape[1]
        for j in range(w):
            sine = (1 - rb_int) + rb_int * np.sin(rb_mult * 3 * j / w)
            input_img[:, j, 1] *= sine
    
    return input_img

def main():
    input_img = cv2.imread("04.png").astype(np.float32) / 255

    start_time = time.process_time()

    yiq = colorRGB2YIQ(input_img)
    yiq = chromaDephase(yiq)
    yiq = rainbowEffect(yiq)
    yiq = blur(yiq)
    bgr = colorYIQRGB(yiq)
    noise_mask = gaussianNoiseMask(bgr.shape, grain_intensity)
    bgr = applyNoiseMask(bgr, noise_mask, grain_amount)    
    bgr2 = scanline(bgr, 0)*bcomp
    bgr2 = np.clip(bgr2, 0, 1)
    bgr = scanline(bgr, 1)*bcomp
    bgr = np.clip(bgr, 0, 1)
    bgr = bloom(bgr.astype(np.float32))
    bgr = darkerBorders(bgr, darkening_strength)
    bgr = curvedBorders(bgr, curve_strength)

    print(f"--- {time.process_time() - start_time:.6f} seconds ---")

    
    '''bgr2 = bloom(bgr2.astype(np.float32))
    bgr2 = darkerBorders(bgr2, darkening_strength)
    bgr2 = curvedBorders(bgr2, curve_strength)
    h, w, c = bgr.shape
    output = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc(*'MJPG'), 60, (w, h))

    total_frames = 600
    bgr = (bgr*255).astype(np.uint8)
    bgr2 = (bgr2*255).astype(np.uint8)
    # Alternar as imagens a cada quadro
    for i in range(total_frames):
        output.write(bgr)
        #output.write(mixed)
        output.write(bgr2)
        #output.write(mixed)

    # Libere o objeto VideoWriter
    output.release()'''
    
    cv2.imshow("final", bgr)
    cv2.waitKey()          
    cv2.imshow("inicial", input_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================