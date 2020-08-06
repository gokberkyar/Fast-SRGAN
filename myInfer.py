from argparse import ArgumentParser
from tensorflow import keras
import json
import numpy as np
import cv2
import os
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--image_dir_Root', type=str, help='Directory where images are kept.')
parser.add_argument('--output_dir_Root', type=str, help='Directory where to output high res images.')
parser.add_argument('--folders', type=str, help='Json file which contains images in each folder')

def main():
    args = parser.parse_args()
    print(args)
    print(args.folders)
    folders = json.load(open(args.folders,"r"))
    print(folders)

    for key in tqdm(folders.keys()):
        image_dir = os.path.join(args.image_dir_Root, key)
        write_dir = os.path.join(args.output_dir_Root, key)
        #print(image_dir)
        #print(write_dir)
        #Get all image paths
        image_paths = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]

        #
        # Change model input shape to accept all size inputs
        model = keras.models.load_model('models/generator.h5')
        inputs = keras.Input((None, None, 3))
        output = model(inputs)
        model = keras.models.Model(inputs, output)

        # Loop over all images
        for image_path in tqdm(image_paths):

            # Read image
            low_res = cv2.imread(image_path, 1)

            # Convert to RGB (opencv uses BGR as default)
            low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)

            # Rescale to 0-1.
            low_res = low_res / 255.0

            # Get super resolution image
            sr = model.predict(np.expand_dims(low_res, axis=0))[0]

            # Rescale values in range 0-255
            sr = ((sr + 1) / 2.) * 255

            # Convert back to BGR for opencv
            sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

            # Save the results:


            cv2.imwrite(os.path.join(write_dir, os.path.basename(image_path)), sr)


if __name__ == '__main__':
    main()
