import cv2 as cv
import os
import ConvolutionKernels as ck



def isValid(filepath):
    return os.path.exists(filepath) and os.path.isfile(filepath)
         

def load_image(filepath):
    print("loading ", filepath)
    return cv.imread(filepath,cv.IMREAD_UNCHANGED)
    


def main(): 
    filepath = "../CVIP/Dataset 0/im2_t.bmp"
    
    if isValid(filepath):
        
        image = load_image(filepath)
        
        sortie = ck.apply(ck.gauss3x3, image)
        
        cv.imshow("Image de Base", image)
        cv.imshow("Image Traitee", sortie)
        
        
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("\nsomething seems wrong with your file path...\n")
main()    