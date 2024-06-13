import numpy as np
import cv2
import math

# constantes de ajuste
ghosting_level = 1
# multiplicador da amplitude (0 = neutro/desliga)  
ghosting_freq = 3
# 2 ou menos desliga
# valores altos pegam frequencias centrais/menores

def hBlending(input_img):
    input_img = cv2.GaussianBlur(input_img, (15, 15), 0)
    return input_img

def ghosting(input_img):
    if ghosting_freq > 2 and ghosting_level > 0:
        base = np.copy(input_img[:, :, 0])
        rows, cols = base.shape
        beta = np.max(base)
        alpha = np.min(base)

        opt_rows = cv2.getOptimalDFTSize(rows)
        opt_cols = cv2.getOptimalDFTSize(cols)
        right = opt_cols - cols
        bottom = opt_rows - rows
        base_padded = cv2.copyMakeBorder(
            base, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=0
        )
        dft = cv2.dft(base_padded, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft = np.fft.fftshift(dft)

        # Máscara multiplicando as frequências altas
        center_col = opt_cols // 2
        center_row = opt_rows // 2
        j_radius = 1 + (opt_cols / ghosting_freq)
        i_radius = 1 + (opt_rows / ghosting_freq)
        mask = np.ones((opt_rows, opt_cols, 2), np.float32)
        for i in range(opt_rows):
            for j in range(opt_cols):
                if abs(i - center_row) > i_radius or abs(j - center_col) > j_radius:
                    mask[i, j] = 0#1 + ghosting_level
        #dft *= mask
        cv2.imshow("input", dft[:, :, 0])
        cv2.waitKey()
        dft_out = np.fft.ifftshift(dft)
        dft_out = cv2.idft(dft_out)
        dft_out = cv2.magnitude(dft_out[:, :, 0], dft_out[:, :, 1])
        # crop no padding
        dft_out = dft_out[:rows, :cols]
        cv2.normalize(dft_out, dft_out, alpha, beta, cv2.NORM_MINMAX)

        input_img[:, :, 0] = dft_out

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

    # bgr **= (1/2.2)

    return bgr


def colorRGB2YIQ(input_img):
    # input_img **= 2.2
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


def main():
    input_img = cv2.imread("02.png").astype(np.float32) / 255

    yiq = colorRGB2YIQ(input_img)
    yiq = ghosting(yiq)
    yiq = hBlending(yiq)
    bgr = colorYIQRGB(yiq)

    cv2.imshow("yiq", input_img)
    cv2.waitKey()
    cv2.imshow("convert", bgr)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================
