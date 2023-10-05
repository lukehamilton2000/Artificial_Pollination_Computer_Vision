import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import glob

frame_index = 1 # Adjust the frame index as needed

for filename in glob.glob("*.*"):
    if '.npy' in filename:
        img_array = np.load(filename, allow_pickle=True)
        
        # Extract a single frame (e.g., the first frame)
        frame = img_array[frame_index]  # Adjust the frame index as needed
        
        plt.imshow(frame, cmap="gray")
        img_name = "index" + str(frame_index) + filename + "_frame.png"   # Adjust the filename as needed
        matplotlib.image.imsave(img_name, frame, cmap="gray")  # Save the single frame as a grayscale image
        print(filename)
