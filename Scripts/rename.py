import os 

# Function to rename multiple files 
def main(): 

    for _, foldername in enumerate(os.listdir("images")): 
        print(type(foldername))

        for count, filename in enumerate(os.listdir("images/" + foldername)): 
            dst =foldername[0] + str(count) + ".jpg"
            src ="images/" + foldername + '/'+ filename 
            dst ="images/" + foldername + '/'+ dst 

            # rename() function will 
            # rename all the files 
            os.rename(src, dst) 

# Driver Code 
if __name__ == '__main__': 

    # Calling main() function 
    main()