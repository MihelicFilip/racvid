import cv2
import numpy as np

 #Pomoč pri implementaciji filtrov je bil Chat-GPT

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
    r1 = np.array([[1,0],[0,-1]]) #2x2 kerneli . r1 je za navpične robove
    r2 = np.array([[0,1],[-1,0]]) #r2 za vodoravne 

    # Get image size
    vr, stolp = slika.shape[:2]

    # Initialize output array
    roberts_amplituda = np.zeros((vr, stolp)) #Naredimo polje nul enake velikosti kot slika

    # Compute the Roberts Cross edges
    for i in range(1, vr-1):  #Gremo skozi vse piksle razen mejnih pikslov zato -1
        for j in range(1, stolp-1):
            r1o = np.sum(r1 * slika[i-1:i+1, j-1:j+1])  #Tu racunamo gradiente z konvolucijo
            r2o = np.sum(r2 * slika[i-1:i+1, j-1:j+1])
            roberts_amplituda[i,j] = np.sqrt(r1o**2 + r2o**2) #Magnituda za trenutni piksel je kvadratni koren vsote gradientov

    # Normalize the magnitude to [0, 1]
    roberts_amplituda /= np.max(roberts_amplituda) #Poskrbimo da ostanejo vrednosti med 0 in 1 
    slika_robov=roberts_amplituda
    return slika_robov

def my_prewitt(slika):
    height, width = slika.shape #Sirina in visina slike 

#3X3 kernel
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) #Detektira vodoravne robove
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) #Detektira navpicne robove 


    slika_robov = np.zeros((height, width), dtype=np.float32) #Naredimo polje nul enake velikosti kot slika

# Apply the Prewitt filter using for loops
    for i in range(1, height - 1): #1 , -1 uporabimo da zajamemo samo piksle, ki so znotraj slike 
        for j in range(1, width - 1):
            fx = np.sum(kernel_x * slika[i - 1 : i + 2, j - 1 : j + 2]) #vodoravne magnitute gradientov 
            fy = np.sum(kernel_y * slika[i - 1 : i + 2, j - 1 : j + 2]) #navpične magnitude gradientov 
            slika_robov[i, j] = np.sqrt(fx**2 + fy**2)

    # Normalizacija izhodne slike 
    slika_robov = cv2.normalize(slika_robov, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    
    return slika_robov



img = cv2.imread("lenna.png",0) 
cv2.imshow("Pokazi sliko ",img)

novaSlika=spremeni_kontrast(img,2,10)

#Roberts algorithm
#roberts = my_roberts(img)
#cv2.imshow("Primerjava roberts ", roberts)

prewittG=my_prewitt(novaSlika)
prewitt=my_prewitt(img)
cv2.imshow("prewitt na spremenjeni sliki",prewittG)
cv2.imshow("prewitt na greyscale",prewitt)




cv2.waitKey()
cv2.destroyAllWindows()