"""Problem Set 4: Motion Detection"""

import os

import cv2
import numpy as np
import sklearn
import ps4

# I/O directories
input_dir = "input_images"
output_dir = "./output_img"

# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in range(0, v.shape[0], stride):

        for x in range(0, u.shape[1], stride):

            cv2.line(img_out, (x, y),
                     (x + int(u[y, x] * scale), y + int(v[y, x] * scale)),
                     color, 1)
            cv2.circle(img_out,
                       (x + int(u[y, x] * scale), y + int(v[y, x] * scale)), 1,
                       color, 1)
    return img_out


# Functions you need to complete:


def scale_u_and_v(u, v, level, pyr):
    """Scales up U and V arrays to match the image dimensions assigned 
    to the first pyramid level: pyr[0].

    You will use this method in part 3. In this section you are asked 
    to select a level in the gaussian pyramid which contains images 
    that are smaller than the one located in pyr[0]. This function 
    should take the U and V arrays computed from this lower level and 
    expand them to match a the size of pyr[0].

    This function consists of a sequence of ps4.expand_image operations 
    based on the pyramid level used to obtain both U and V. Multiply 
    the result of expand_image by 2 to scale the vector values. After 
    each expand_image operation you should adjust the resulting arrays 
    to match the current level shape 
    i.e. U.shape == pyr[current_level].shape and 
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Hint: create a for loop from level-1 to 0 inclusive.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:
        u: U array obtained from ps4.optic_flow_lk
        v: V array obtained from ps4.optic_flow_lk
        level: level value used in the gaussian pyramid to obtain U 
               and V (see part_3)
        pyr: gaussian pyramid used to verify the shapes of U and V at 
             each iteration until the level 0 has been met.

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to 
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to 
                             pyr[0].shape
    """
    for i in range(level-1,-1,-1):
        u_exp = ps4.expand_image(u)
        v_exp = ps4.expand_image(v)
        u = u_exp[0:np.shape(pyr[i])[0], 0:np.shape(pyr[i])[1]]
        v = v_exp[0:np.shape(pyr[i])[0], 0:np.shape(pyr[i])[1]]
    return u, v

def part_1a():

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'),
                         0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR2.png'),
                          0) / 255.
    shift_r5_u5 = cv2.imread(
        os.path.join(input_dir, 'TestSeq', 'ShiftR5U5.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    k_size = 25  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    # input images first.

    # k_size = 60  # TODO: Select a kernel size
    # k_type = "uniform"  # TODO: Select a kernel type
    # sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
    # u, v = ps4.optic_flow_lk(shift_0, shift_r5_u5, k_size, k_type, sigma)

    k_size = 45  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 45  # TODO: Select a sigma value if you are using a gaussian kernel

    shift_0_blur = cv2.GaussianBlur(shift_0, ksize=(k_size, k_size), sigmaX=sigma, sigmaY=sigma,
                                    borderType=cv2.BORDER_DEFAULT)
    shift_r5_u5_blur = cv2.GaussianBlur(shift_r5_u5, ksize=(k_size, k_size), sigmaX=sigma, sigmaY=sigma,
                                    borderType=cv2.BORDER_DEFAULT)

    u, v = ps4.optic_flow_lk(shift_0_blur, shift_r5_u5_blur, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)


def part_1b():
    """Performs the same operations applied in part_1a using the images
    ShiftR10, ShiftR20 and ShiftR40.

    You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Make sure you explore different parameters and/or pre-process the
    input images to improve your results.

    In this part you should save the following images:
    - ps4-1-b-1.png
    - ps4-1-b-2.png
    - ps4-1-b-3.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'),
                         0) / 255.

    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'),
                           0) / 255.

    k_size = 65  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 65  # TODO: Select a sigma value if you are using a gaussian kernel
    shift_0_blur = cv2.GaussianBlur(shift_0, ksize=(k_size, k_size), sigmaX=sigma, sigmaY=sigma,
                                    borderType=cv2.BORDER_DEFAULT)
    shift_r10_blur = cv2.GaussianBlur(shift_r10, ksize=(k_size, k_size), sigmaX=sigma, sigmaY=sigma,
                                    borderType=cv2.BORDER_DEFAULT)

    u, v = ps4.optic_flow_lk(shift_0_blur, shift_r10_blur, k_size, k_type, sigma)
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-1.png"), u_v)

    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'),
                           0) / 255.

    k_size = 71  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 71  # TODO: Select a sigma value if you are using a gaussian kernel
    shift_0_blur = cv2.GaussianBlur(shift_0, ksize=(k_size, k_size), sigmaX=sigma, sigmaY=sigma,
                                    borderType=cv2.BORDER_DEFAULT)
    shift_r20_blur = cv2.GaussianBlur(shift_r20, ksize=(k_size, k_size), sigmaX=sigma, sigmaY=sigma,
                                      borderType=cv2.BORDER_DEFAULT)
    u, v = ps4.optic_flow_lk(shift_0_blur, shift_r20_blur, k_size, k_type, sigma)
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-2.png"), u_v)



    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'),
                           0) / 255.
    k_size = 91  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 91  # TODO: Select a sigma value if you are using a gaussian kernel
    shift_0_blur = cv2.GaussianBlur(shift_0, ksize=(k_size, k_size), sigmaX=sigma, sigmaY=sigma,
                                    borderType=cv2.BORDER_DEFAULT)
    shift_r40_blur = cv2.GaussianBlur(shift_r40, ksize=(k_size, k_size), sigmaX=sigma, sigmaY=sigma,
                                      borderType=cv2.BORDER_DEFAULT)
    u, v = ps4.optic_flow_lk(shift_0_blur, shift_r40_blur, k_size, k_type, sigma)
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-3.png"), u_v)



def part_2():

    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-a-1.png"),
                yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)

    yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-b-1.png"),
                yos_img_01_l_pyr_img)


def part_3a_1():
    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    levels = 1  # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)

    level_id = 0  # TODO: Select the level number (or id) you wish to use
    k_size = 45  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 45  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
                             yos_img_02_g_pyr[level_id], k_size, k_type, sigma)

    u = np.reshape(u,(np.shape(u)[0],np.shape(u)[1]))
    v = np.reshape(v,(np.shape(v)[0],np.shape(v)[1]))

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
                ps4.normalize_and_scale(diff_yos_img_01_02))


def part_3a_2():
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    levels = 10  # Define the number of pyramid levels
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
    yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)

    level_id = 0  # TODO: Select the level number (or id) you wish to use
    k_size = 15  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 15  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
                             yos_img_03_g_pyr[level_id], k_size, k_type, sigma)

    u = np.reshape(u,(np.shape(u)[0],np.shape(u)[1]))
    v = np.reshape(v,(np.shape(v)[0],np.shape(v)[1]))

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)

    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_4a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'),
                         0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'),
                           0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'),
                           0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'),
                           0) / 255.

    levels = 5  # TODO: Define the number of levels
    k_size = 21  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 21  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)



    u_v = quiver(u10, v10, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)

    levels = 4  # TODO: Define the number of levels
    k_size = 71  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 45  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    # You may want to try different parameters for the remaining function
    # calls.
    u20, v20 = ps4.hierarchical_lk(shift_0, shift_r20, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u20, v20, scale=10, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)


    levels = 5  # TODO: Define the number of levels
    k_size = 71  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 71  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values


    u40, v40 = ps4.hierarchical_lk(shift_0, shift_r40, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    u_v = quiver(u40, v40, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)


# alpha_slider_max = 100
# title_window = 'Linear Blend'

# def on_trackbar(val):
#     alpha = val / alpha_slider_max
#     beta = ( 1.0 - alpha )
#     dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
#     cv2.imshow(title_window, dst)

def part_4b():
    urban_img_01 = cv2.imread(os.path.join(input_dir, 'Urban2', 'urban01.png'),
                              0) / 255.
    urban_img_02 = cv2.imread(os.path.join(input_dir, 'Urban2', 'urban02.png'),
                              0) / 255.

    levels = 7  # TODO: Define the number of levels
    k_size = 11  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = k_size  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    u_v = quiver(u, v, scale=1, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    diff_img = urban_img_01 - urban_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
                ps4.normalize_and_scale(diff_img))

    # trackbar_name = 'Alpha x %d' % alpha_slider_max
    # cv2.createTrackbar(trackbar_name, title_window, 0, alpha_slider_max, on_trackbar)

def part_5a():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'),
                         0) / 255.

    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'),
                           0) / 255.
    levels = 4
    k_size = 13 # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = k_size  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    u, v = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    new_img_count = 4
    new_img_list = []
    new_img_list.append(shift_0)
    cv2.imwrite(os.path.join(output_dir, "part_5a_" + str(0) + ".png"), ps4.normalize_and_scale(shift_0))
    cv2.imwrite(os.path.join(output_dir, "part_5a_" + str(new_img_count+1) + ".png"), ps4.normalize_and_scale(shift_r10))

    for i in range(new_img_count):
        partial_u = u * (new_img_count-i+1)/(new_img_count+1)
        partial_v = v * (new_img_count-i+1)/(new_img_count+1)
        frame = ps4.warp(shift_r10,partial_u,partial_v,interpolation,border_mode)
        cv2.imwrite(os.path.join(output_dir, "part_5a_" + str(i+1) +".png"), ps4.normalize_and_scale(frame))
        new_img_list.append(frame)

    new_img_list.append(shift_r10)
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "part_5a quiver.png"), u_v)

    # create combo img
    combo_img = np.vstack((np.hstack((new_img_list[0],new_img_list[1],new_img_list[2])),np.hstack((new_img_list[3],new_img_list[4],new_img_list[5]))))
    # cv2.imshow('combo_img',combo_img)
    # cv2.waitKey()

    cv2.imwrite(os.path.join(output_dir, "ps4-5-a-1.png"),ps4.normalize_and_scale(combo_img))
def part_5b():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    img_a = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc01.png'),
                         0) / 255.

    img_b = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc02.png'),
                           0) / 255.

    img_c = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc03.png'),
                           0) / 255.
    levels = 6
    k_size = 25  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = k_size  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    u, v = ps4.hierarchical_lk(img_a, img_b, levels, k_size,
                               k_type, sigma, interpolation, border_mode)


    #Part a
    #threshold u and v using u,v magnitude
    # mag1 = np.square(u)+np.square(v)
    # mag2 = np.where(mag1<1,0,mag1)
    # median = np.median(np.where(mag2>0))
    # mag3 = np.where(mag2 > 4  * median,0,mag2 )
    # u = np.where(mag3 != 0,u,0)
    # v = np.where(mag3 != 0, v, 0)

    new_img_count = 4
    new_img_list = []
    new_img_list.append(img_a)
    cv2.imwrite(os.path.join(output_dir, "part_5bi_" + str(0) + ".png"), ps4.normalize_and_scale(img_a))
    cv2.imwrite(os.path.join(output_dir, "part_5bi_" + str(new_img_count + 1) + ".png"),
                ps4.normalize_and_scale(img_b))
    for i in range(new_img_count):
        partial_u = u * (new_img_count - i) / (new_img_count + 1)
        partial_v = v * (new_img_count - i) / (new_img_count + 1)
        frame = ps4.warp(img_b, partial_u, partial_v, interpolation, border_mode)
        cv2.imwrite(os.path.join(output_dir, "part_5bi_" + str(i + 1) + ".png"), ps4.normalize_and_scale(frame))
        new_img_list.append(frame)
    new_img_list.append(img_b)
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "part_5bi quiver.png"), u_v)

    # create combo img
    combo_img = np.vstack((np.hstack((new_img_list[0], new_img_list[1], new_img_list[2])),
                           np.hstack((new_img_list[3], new_img_list[4], new_img_list[5]))))
    cv2.imwrite(os.path.join(output_dir, "ps4-5-b-1.png"), ps4.normalize_and_scale(combo_img))

    #Part b
    levels = 5
    k_size = 25  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = k_size  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_LINEAR  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u, v = ps4.hierarchical_lk(img_b,img_c,  levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    #threshold u and v using u,v magnitude
    # mag1 = np.square(u)+np.square(v)
    # mag2 = np.where(mag1<1,0,mag1)
    # median = np.median(np.where(mag2>0))
    # mag3 = np.where(mag2 > 4  * median,0,mag2 )
    # u = np.where(mag3 != 0,u,0)
    # v = np.where(mag3 != 0, v, 0)

    new_img_count = 4
    new_img_list = []
    new_img_list.append(img_b)
    cv2.imwrite(os.path.join(output_dir, "part_5bii_" + str(0) + ".png"), ps4.normalize_and_scale(img_b))
    cv2.imwrite(os.path.join(output_dir, "part_5bii_" + str(new_img_count + 1) + ".png"),
                ps4.normalize_and_scale(img_c))
    for i in range(new_img_count):
        # partial_u = u * (new_img_count - i) / (new_img_count + 1)
        # partial_v = v * (new_img_count - i) / (new_img_count + 1)
        partial_u = 0.2 * (i+1.) * u
        partial_v = 0.2 * (i+1.) * v
        frame = ps4.warp(img_b, partial_u, partial_v, interpolation, border_mode)
        cv2.imwrite(os.path.join(output_dir, "part_5bii_" + str(i + 1) + ".png"), ps4.normalize_and_scale(frame))
        new_img_list.append(frame)
        # w_img = cv2.addWeighted(frame,(new_img_count - i) / (new_img_count + 1),img_c,
        #                         1- (new_img_count - i) / (new_img_count + 1),0)
        # cv2.imwrite(os.path.join(output_dir, "part_5bii_wimg" + str(i + 1) + ".png"), ps4.normalize_and_scale(w_img))

    new_img_list.append(img_c)
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "part_5bii quiver.png"), u_v)

    # create combo img
    combo_img = np.vstack((np.hstack((new_img_list[0], new_img_list[1], new_img_list[2])),
                           np.hstack((new_img_list[3], new_img_list[4], new_img_list[5]))))
    cv2.imwrite(os.path.join(output_dir, "ps4-5-b-2.png"), ps4.normalize_and_scale(combo_img))


def part_6():

    """Challenge Problem

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    video_list = ['person01_running_d2_uncomp.avi',
                  'person04_walking_d4_uncomp.avi',
                  'person06_walking_d1_uncomp.avi',
                  'person08_running_d4_uncomp.avi',
                  'person10_handclapping_d4_uncomp.avi',
                  'person14_handclapping_d3_uncomp.avi']

    for video in video_list:
        path = os.path.join(input_dir, 'videos', video)
        frames = ps4.read_video(path)
        print(video,'- Class of video: ', ps4.classify_video(frames))



if __name__ == '__main__':
    # part_1a()
    # part_1b()
    # part_2()
    # part_3a_1()
    # part_3a_2()
    # part_4a()
    # part_4b()
    # part_5a()
    part_5b()
    # part_6()
