# This file has NOTHING to do with the rest of the MP
# Please answer the filter question in your report with
# results from running this script
# You still have to do "source devel/setup.bash".
# Eric Liang

import os
import cv2

# Gaussian Filter Function
def filter_gaussian(input_img):
    img = input_img.copy()
    # Apply Gaussian Blur with a 5x5 kernel
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

# Median Filter Function
def filter_median(input_img):
    img = input_img.copy()
    # Apply Median Blur with a kernel size of 5
    img = cv2.medianBlur(img, 5)
    return img

if __name__ == '__main__':
    # Set the image paths directly (update these with your actual paths)
    sp_img_path = '/Users/xueqingli/Downloads/Quad-Squad-Works/src/mp1/images/salt_and_pepper.jpg'
    ga_img_path = '/Users/xueqingli/Downloads/Quad-Squad-Works/src/mp1/images/gaussian.jpg'

    # Set the result image paths (update these with your actual paths)
    result_sp_img_ga_filter_path = '/Users/xueqingli/Downloads/Quad-Squad-Works/src/mp1/images/salt_and_pepper_gaussian_filter.jpg'
    result_sp_img_me_filter_path = '/Users/xueqingli/Downloads/Quad-Squad-Works/src/mp1/images/salt_and_pepper_median_filter.jpg'
    result_ga_img_ga_filter_path = '/Users/xueqingli/Downloads/Quad-Squad-Works/src/mp1/images/gaussian_noise_gaussian_filter.jpg'
    result_ga_img_me_filter_path = '/Users/xueqingli/Downloads/Quad-Squad-Works/src/mp1/images/gaussian_noise_median_filter.jpg'

    # Read the images
    sp_img = cv2.imread(sp_img_path)
    ga_img = cv2.imread(ga_img_path)

    # Run the filters on the salt-and-pepper noise image
    sp_img_gaussian_filter = filter_gaussian(sp_img)
    sp_img_median_filter = filter_median(sp_img)

    # Run the filters on the Gaussian noise image
    ga_img_gaussian_filter = filter_gaussian(ga_img)
    ga_img_median_filter = filter_median(ga_img)

    # Show the images
    cv2.imshow("Salt and Pepper before Filtering", sp_img)
    cv2.imshow("Salt and Pepper after Gaussian Filter", sp_img_gaussian_filter)
    cv2.imshow("Salt and Pepper after Median Filter", sp_img_median_filter)
    cv2.imshow("Gaussian Noise before Filtering", ga_img)
    cv2.imshow("Gaussian Noise after Gaussian Filter", ga_img_gaussian_filter)
    cv2.imshow("Gaussian Noise after Median Filter", ga_img_median_filter)

    # Write the filtered images to the results folder
    cv2.imwrite(result_sp_img_ga_filter_path, sp_img_gaussian_filter)
    cv2.imwrite(result_sp_img_me_filter_path, sp_img_median_filter)
    cv2.imwrite(result_ga_img_ga_filter_path, ga_img_gaussian_filter)
    cv2.imwrite(result_ga_img_me_filter_path, ga_img_median_filter)

    # Pause to show the images
    print("Press any key on image windows to quit")
    cv2.waitKey(0)
