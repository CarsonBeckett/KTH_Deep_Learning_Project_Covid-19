import os
import shutil
import glob
import random


num_rand_exp = 5
num_images = 50

# Function to rename multiple files 
def main(): 

    for count in range(num_rand_exp):

        # to_be_moved = random.sample(glob.glob("C:/Users/Alonso/Desktop/ACE/google-images-deep-learning/images/_All/*.jpg"), 200)
        # to_be_moved = random.sample(glob.glob("C:/Users/Alonso/Desktop/google-images-deep-learning/flower_photos/_all/*.jpg"), 200)
        to_be_moved = random.sample(glob.glob("C:/Users/Alonso/Desktop/DL/covid_dataset/_all/*.*"), num_images)

        for f in enumerate(to_be_moved, 1):
            dest = "C:/Users/Alonso/Desktop/ACE/random500_" + str(count) + "/"
            # dest = "C:/Users/Alonso/Desktop/ACE/random_discovery/" 
            if not os.path.exists(dest):
                os.makedirs(dest)
            shutil.copy(f[1], dest)

# Driver Code 
if __name__ == '__main__': 

    # Calling main() function 
    main()