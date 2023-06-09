import cv2
import numpy as np

 #Pomoč pri implementaciji filtrov je bil Chat-GPT

def Overlay(slikaOrg, Robovi):
    slika_robov = np.zeros((slikaOrg.shape[0], slikaOrg.shape[1], 3), np.uint8)
    slikaBarva = cv2.cvtColor(slikaOrg, cv2.COLOR_GRAY2BGR)
    slika_robov[:, :, 1] = Robovi
    slika_robov = cv2.addWeighted(slikaBarva, 0.4, slika_robov, 1.4, 3)
    return slika_robov

def spremeni_kontrast(slika, alfa, beta):
    spremenjena_slika = alfa * slika + beta
    spremenjena_slika = np.clip(spremenjena_slika, 0, 255)
    spremenjena_slika = spremenjena_slika.astype(np.uint8)
    return spremenjena_slika


def my_roberts(slika):
    r1 = np.array([[1,0],[0,-1]]) #2x2 kerneli . r1 je za navpične robove
    r2 = np.array([[0,1],[-1,0]]) #r2 za vodoravne 

    vr, stolp = slika.shape[:2]

    roberts_amplituda = np.zeros((vr, stolp)) #Naredimo polje nul enake velikosti kot slika

    for i in range(1, vr-1):  #Gremo skozi vse piksle razen mejnih pikslov zato -1
        for j in range(1, stolp-1):
            r1o = np.sum(r1 * slika[i-1:i+1, j-1:j+1])  #Tu racunamo gradiente z konvolucijo
            r2o = np.sum(r2 * slika[i-1:i+1, j-1:j+1])
            roberts_amplituda[i,j] = np.sqrt(r1o**2 + r2o**2) #Magnituda za trenutni piksel je kvadratni koren vsote gradientov

    roberts_amplituda /= np.max(roberts_amplituda) #Poskrbimo da ostanejo vrednosti med 0 in 1 
    slika_robov=roberts_amplituda

    return slika_robov

def my_prewitt(slika):
    height, width = slika.shape #Sirina in visina slike 


    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) #Detektira vodoravne robove
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) #Detektira navpicne robove 


    slika_robov = np.zeros((height, width), dtype=np.float32) #Naredimo polje nul enake velikosti kot slika

    for i in range(1, height - 1): #1 , -1 uporabimo da zajamemo samo piksle, ki so znotraj slike 
        for j in range(1, width - 1):
            fx = np.sum(kernel_x * slika[i - 1 : i + 2, j - 1 : j + 2]) #vodoravne magnitute gradientov 
            fy = np.sum(kernel_y * slika[i - 1 : i + 2, j - 1 : j + 2]) #navpične magnitude gradientov 
            slika_robov[i, j] = np.sqrt(fx**2 + fy**2)

    # Normalizacija izhodne slike 
    slika_robov = cv2.normalize(slika_robov, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    
    return slika_robov

def my_sobel(slika):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) #horizontalni robovi
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) #vertikalni robovi

    
    height, width = slika.shape

    
    slika_robov = np.zeros((height, width)) # Slika v katero shranjujemo magnitude gradientov 

   
    for i in range(1, height - 1):
        for j in range(1, width - 1): #Gremo skozi vse piksle in uporabimo Euclideanovo enacbo za racunanje magnitude gradientov 
            gx = (sobel_x[0][0] * slika[i-1][j-1]) + (sobel_x[0][1] * slika[i-1][j]) + \
                 (sobel_x[0][2] * slika[i-1][j+1]) + (sobel_x[1][0] * slika[i][j-1]) + \
                 (sobel_x[1][1] * slika[i][j]) + (sobel_x[1][2] * slika[i][j+1]) + \
                 (sobel_x[2][0] * slika[i+1][j-1]) + (sobel_x[2][1] * slika[i+1][j]) + \
                 (sobel_x[2][2] * slika[i+1][j+1])

            gy = (sobel_y[0][0] * slika[i-1][j-1]) + (sobel_y[0][1] * slika[i-1][j]) + \
                 (sobel_y[0][2] * slika[i-1][j+1]) + (sobel_y[1][0] * slika[i][j-1]) + \
                 (sobel_y[1][1] * slika[i][j]) + (sobel_y[1][2] * slika[i][j+1]) + \
                 (sobel_y[2][0] * slika[i+1][j-1]) + (sobel_y[2][1] * slika[i+1][j]) + \
                 (sobel_y[2][2] * slika[i+1][j+1])

            slika_robov[i][j] = np.sqrt(gx**2 + gy**2) #Izracunamo magnitudo gradienta za vsak pixel na i,j poziciji

    slika_robov = cv2.normalize(slika_robov, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return slika_robov


def canny(slika, sp_prag, zg_prag):
    slika_robov=cv2.Canny(slika,sp_prag,zg_prag)
    return slika_robov 


img = cv2.imread("lenna.png",0) 
cv2.imshow("Pokazi sliko ",img)

novaSlika=spremeni_kontrast(img,0.5,10)

#Roberts algorithm
#roberts = my_roberts(img)
#cv2.imshow("Roberts ",roberts)

#lol=merge_images(img,roberts)

#cv2.imshow("Roboviiii",lol)

# show the result
#prewittG=my_prewitt(novaSlika)
prewitt=my_prewitt(novaSlika)
#cv2.imshow("prewitt",prewitt)

#sobel = my_sobel(novaSlika)
#cv2.imshow("Sobel",sobel)

Lower=20
Upper=70
Can=canny(novaSlika,10,100)
cv2.imshow("canny filter",prewitt)

over=Overlay(img,prewitt)
cv2.imshow("overlayed edges",over)
Gauss= cv2.GaussianBlur(novaSlika,(5,5),10)
cv2.imshow("Gauss ",Gauss)
prewitt2=my_prewitt(Gauss)
cv2.imshow("Gauss na sliki pred prewitt",prewitt2)

cv2.waitKey()
cv2.destroyAllWindows()