import os
import cv2
import numpy as np
from glob import glob
import csv
from histomicstk.cli.utils import CLIArgumentParser

def main(args):
    descriptor_list = []

    #
    # read input regions and masks
    #
    print('>> Reading regions and masks ..')

    region_path = args.inputDirectory + '/region'
    mask_path = args.inputDirectory + '/mask'
    isdir_region = os.path.isdir(region_path)
    isdir_mask = os.path.isdir(mask_path)
    if isdir_region != True and isdir_mask != True:
        raise ValueError('Corresponding directory not found')

    imgs = glob(region_path + '/' +'*' + '.png')
    if imgs == None:
        raise ValueError('No images found in the specified directory')
    for idx,img in enumerate(imgs):
        filename = os.path.basename(img)
        mask = mask_path + '/' + os.path.splitext(filename)[0] + '.png'
        isfile = os.path.exists(mask)
        if isfile != True:
            raise ValueError('Corresponding mask not found')
        image_bgr = cv2.imread(img)
        mask_bgr = cv2.imread(mask)
        src2 = cv2.resize(mask_bgr, image_bgr.shape[1::-1])
        dst = cv2.bitwise_and(image_bgr, src2)
        (means, stds) = cv2.meanStdDev(dst)
        stats = np.concatenate([means, stds]).flatten()
        stats = stats.tolist()
        d = {}
        d[filename] = stats
        descriptor_list.append(d)
        #result = os.path.dirname(region_path) + '/' + filename + '.png'
        #cv2.imwrite(result, dst)

    #
    # save file in .csv
    #
    print('>> Save the color descriptor in a file ..')

    resultPath = args.outputDescriptorFile
    desFile = open(resultPath, 'w')
    with desFile:
        desFields = ['annoation', 'mean-blue', 'mean-green', 'mean-red', 'std-blue','std-green','std-red']
        writer = csv.DictWriter(desFile, fieldnames=desFields)
        writer.writeheader()
        for item in descriptor_list:
            for key in item:
                val = item[key]
                writer.writerow({'annoation' : key, 'mean-blue': val[0], 'mean-green' : val[1], 'mean-red': val[2],
                                 'std-blue' : val[3], 'std-green' : val[4], 'std-red': val[5]})

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())