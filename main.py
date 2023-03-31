import cv2
import numpy as np

 #Pri implementaciji algoritmov je bil uporabljen Chat-GPT

def spremeni_kontrast(slika, alfa, beta):
    # Scale the pixel values based on the contrast and brightness
    spremenjena_slika = alfa * slika + beta
    # Clip the pixel values to ensure they stay within the valid range
    spremenjena_slika = np.clip(spremenjena_slika, 0, 255)
    # Convert the adjusted NumPy array back to a 2D NumPy array and return it
    spremenjena_slika = spremenjena_slika.astype(np.uint8)
    return spremenjena_slika


def my_roberts(slika):
# Define the horizontal and vertical Roberts Cross kernels
    r1 = np.array([[1,0],[0,-1]])
    r2 = np.array([[0,1],[-1,0]])

    # Get image size
    vr, stolp = slika.shape[:2]

    # Initialize output array
    roberts_amplituda = np.zeros((vr, stolp))

    # Compute the Roberts Cross edges
    for i in range(1, vr):
        for j in range(1, stolp):
            r1o = np.sum(r1 * slika[i-1:i+1, j-1:j+1])
            r2o = np.sum(r2 * slika[i-1:i+1, j-1:j+1])
            roberts_amplituda[i,j] = np.sqrt(r1o**2 + r2o**2)

    # Normalize the magnitude to [0, 1]
    roberts_amplituda /= np.max(roberts_amplituda)
    slika_robov=roberts_amplituda
    return slika_robov


img = cv2.imread("lenna.png",0) 
cv2.imshow("Pokazi sliko ",img)

novaSlika=spremeni_kontrast(img,3,5)

#Roberts algorithm
roberts = my_roberts(img)
cv2.imshow("Primerjava roberts ", roberts)


cv2.waitKey()
cv2.destroyAllWindows()