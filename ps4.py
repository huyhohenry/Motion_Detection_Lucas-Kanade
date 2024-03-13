
import cv2 as cv2
import numpy as np


# Utility function
import scipy.signal


def read_video(video_file, show=False):
    """Reads a video file and outputs a list of consecuative frames
  Args:
      image (string): Video file path
      show (bool):    Visualize the input video. WARNING doesn't work in
                      notebooks
  Returns:
      list(numpy.ndarray): list of frames
  """
    frames = []
    cap = cv2.VideoCapture(video_file)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        # Opens a new window and displays the input
        if show:
            cv2.imshow("input", frame)
            # Frames are read by intervals of 1 millisecond. The
            # programs breaks out of the while loop when the
            # user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # The following frees up resources and
    # closes all windows
    cap.release()
    if show:
        cv2.destroyAllWindows()
    return frames
    
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """

    grandient_x = cv2.Sobel(image,ddepth = cv2.CV_64F, dx=1, dy=0, ksize=3, scale =1/8,borderType=cv2.BORDER_DEFAULT)
    return grandient_x


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """

    grandient_y = cv2.Sobel(image,ddepth = cv2.CV_64F,dx=0, dy=1, ksize=3, scale =1/8,borderType=cv2.BORDER_DEFAULT )
    return grandient_y


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    Ix = gradient_x(img_a)
    Iy = gradient_y(img_a)
    It = img_b - img_a
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    Ixt = Ix * It
    Iyt = Iy * It
    # Apply filter to 5 gradients above:

    if k_type == 'uniform':
        Sxx = cv2.boxFilter(Ixx,ddepth = cv2.CV_64F,ksize = (k_size,k_size),borderType=cv2.BORDER_DEFAULT)
        Syy = cv2.boxFilter(Iyy,ddepth = cv2.CV_64F,ksize = (k_size,k_size),borderType=cv2.BORDER_DEFAULT)
        Sxy = cv2.boxFilter(Ixy,ddepth = cv2.CV_64F,ksize = (k_size,k_size),borderType=cv2.BORDER_DEFAULT)
        Sxt = cv2.boxFilter(Ixt,ddepth = cv2.CV_64F,ksize = (k_size,k_size),borderType=cv2.BORDER_DEFAULT)
        Syt = cv2.boxFilter(Iyt,ddepth = cv2.CV_64F,ksize = (k_size,k_size),borderType=cv2.BORDER_DEFAULT)
    else:
        Sxx = cv2.GaussianBlur(Ixx,ksize=(k_size,k_size), sigmaX= sigma, sigmaY=sigma,borderType=cv2.BORDER_DEFAULT)
        Syy = cv2.GaussianBlur(Iyy,ksize=(k_size,k_size), sigmaX= sigma, sigmaY=sigma,borderType=cv2.BORDER_DEFAULT)
        Sxy = cv2.GaussianBlur(Ixy,ksize=(k_size,k_size), sigmaX= sigma, sigmaY=sigma,borderType=cv2.BORDER_DEFAULT)
        Sxt = cv2.GaussianBlur(Ixt,ksize=(k_size,k_size), sigmaX= sigma, sigmaY=sigma,borderType=cv2.BORDER_DEFAULT)
        Syt = cv2.GaussianBlur(Iyt,ksize=(k_size,k_size), sigmaX= sigma, sigmaY=sigma,borderType=cv2.BORDER_DEFAULT)

    #reshape for broadcasting
    Sxx_r = Sxx.reshape(np.shape(img_a) + (1,))
    Syy_r = Syy.reshape(np.shape(img_a) + (1,))
    Sxy_r = Sxy.reshape(np.shape(img_a) + (1,))
    Sxt_r = Sxt.reshape(np.shape(img_a) + (1,))
    Syt_r = Syt.reshape(np.shape(img_a) + (1,))

    # A * d = b
    # d = b * A_inv
    A_inv = np.zeros(np.shape(img_a) + (4,))
    b = np.zeros(np.shape(img_a) + (2,))

    det = np.where(Sxx_r * Syy_r - Sxy_r * Sxy_r !=0,Sxx_r * Syy_r - Sxy_r * Sxy_r,0)
    # a:Sxx_r
    # b:Sxy_r
    # c:Sxy_r
    # d:Syy_r
    A_inv[..., [0]] = np.where(det !=0,Syy_r / det,0)
    A_inv[..., [1]] = np.where(det !=0,-Sxy_r / det,0)
    A_inv[..., [2]] = np.where(det !=0,-Sxy_r / det,0)
    A_inv[..., [3]] = np.where(det !=0,Sxx_r / det,0)

    b[..., [0]] = -Sxt_r
    b[..., [1]] = -Syt_r

    A_inv_reshape = np.reshape(A_inv,(np.shape(A_inv)[0],np.shape(A_inv)[1],2,2))
    b_reshape = np.reshape(b,(np.shape(b)[0],np.shape(b)[1],2,1))
    uv =  np.matmul(A_inv_reshape, b_reshape)
    u = uv[:,:,0]
    v = uv[:, :, 1]
    return (u,v)


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """

    filtered_img = cv2.sepFilter2D(src = image,ddepth = cv2.CV_64F,kernelX=(1/16,4/16,6/16,4/16,1/16),
                                  kernelY=(1/16,4/16,6/16,4/16,1/16),borderType=cv2.BORDER_DEFAULT)

    row_reduced = np.delete(filtered_img,np.arange(1,np.shape(filtered_img)[0],2),0)
    col_reduced = np.delete(row_reduced,np.arange(1,np.shape(row_reduced)[1],2),1)

    return col_reduced



def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    gaussian_pyramid = []
    gaussian_pyramid.append(image)
    for i in range(1,levels):
        gaussian_pyramid.append(reduce_image(gaussian_pyramid[i-1]))
    return gaussian_pyramid

def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    combined_img = normalize_and_scale(img_list[0])
    for i in range(1,len(img_list)):
        # normalize image
        norm_img = normalize_and_scale(img_list[i])
        # extend img_list[i] to have the same number of row with 1st image
        padded_img = np.pad(norm_img,[(0, np.shape(combined_img)[0] - np.shape(norm_img)[0]),(0,0)],mode = 'constant' )
        combined_img = np.hstack((combined_img,padded_img))
    return combined_img


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    #inflate image by adding zero row and columns in between image row/column
    n = 1
    image_inflated = np.zeros((n + 1) * np.array(image.shape), dtype=image.dtype)
    expanded_image = np.zeros((n + 1) * np.array(image.shape), dtype=image.dtype)
    image_inflated[::n + 1, ::n + 1] = image

    # # run 2dsepfilter
    filtered_img = cv2.sepFilter2D(src = image_inflated,ddepth = cv2.CV_64F,kernelX=(2/16,8/16,12/16,8/16,2/16),
                                  kernelY=(2/16,8/16,12/16,8/16,2/16),borderType=cv2.BORDER_DEFAULT)

    return filtered_img



def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    g_pyr = np.flip(g_pyr)
    l_pyr = g_pyr.copy()

    for i in range(1,len(g_pyr)):
        exp_img = expand_image(g_pyr[i-1])
        exp_img_trim = exp_img[0:np.shape(g_pyr[i])[0],0:np.shape(g_pyr[i])[1]]
        l_pyr[i] = g_pyr[i] - exp_img_trim
    l_pyr = np.flip(l_pyr)
    return list(l_pyr)


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    # img_use = normalize_and_scale(image)
    img_use = image.copy()
    h, w = np.shape(img_use)

    # index_x, index_y = np.indices((h, w), dtype=np.float64)
    index_y, index_x = np.indices((h, w), dtype=np.float64)
    map_y = index_y + V
    map_y = map_y.astype(np.float32)
    map_x = index_x + U
    map_x = map_x.astype(np.float32)
    img_warp = cv2.remap(img_use, map_x,map_y, interpolation=interpolation,borderMode=border_mode)
    # print()
    return  img_warp

# if __name__ == "__main__":
#     image = cv2.imread("./input_images/DataSeq1/yos_img_01.jpg", 0)/255.
#     U = np.ones(np.shape(image))
#     V = np.ones(np.shape(image))
#     imgwarp = warp(image, U=U, V=V, interpolation = cv2.INTER_CUBIC, border_mode = cv2.BORDER_REFLECT101)


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """
    #reduce both image to levels as specified
    gaus_pyr_a = gaussian_pyramid(img_a, levels=levels)
    gaus_pyr_b = gaussian_pyramid(img_b, levels=levels)
    u, v = optic_flow_lk(gaus_pyr_a[levels-1], gaus_pyr_b[levels-1], k_size, k_type, sigma)
    u = np.reshape(u, (np.shape(u)[0], np.shape(u)[1]))
    v = np.reshape(v, (np.shape(v)[0], np.shape(v)[1]))
    for i in range(levels-1,0,-1):

        u = expand_image(u)
        v = expand_image(v)
        u = u[0:np.shape(gaus_pyr_b[i-1])[0], 0:np.shape(gaus_pyr_b[i-1])[1]]
        v = v[0:np.shape(gaus_pyr_b[i-1])[0], 0:np.shape(gaus_pyr_b[i-1])[1]]
        u = np.reshape(u, (np.shape(u)[0], np.shape(u)[1])) * 2
        v = np.reshape(v, (np.shape(v)[0], np.shape(v)[1])) * 2

        #warp img b torward img a
        b_warp = warp(gaus_pyr_b[i-1],u,v,interpolation,border_mode)
        u_prime,v_prime = optic_flow_lk(gaus_pyr_a[i-1],b_warp,k_size,k_type,sigma)
        u_prime = np.reshape(u_prime, (np.shape(u_prime)[0], np.shape(u_prime)[1]))
        v_prime = np.reshape(v_prime, (np.shape(v_prime)[0], np.shape(v_prime)[1]))

        u = np.add(u,u_prime)
        v = np.add(v,v_prime)


    return u,v

def classify_video(images):
    """Classifies a set of frames as either
    - int(1) == "Running"
    - int(2) == "Walking"
    - int(3) == "Clapping"
    Args:
      images list(numpy.array): greyscale floating-point frames of a video
    Returns:
      int:  Class of video
    """
    # process n seconds of the videos - 25 frames per second
    n_sec = 3

    # position of 1st frame to process - to filter out first few frames without any actions
    f_frame = 8

    # number_of_frame_to_process - for 25 fps videos
    n_frame = n_sec * 25

    #cummulative sum number of moving pixels in all frames
    moving_pixel_sum = 0

    #cummulative sum of displacement vector magnitude
    displacement_sum = 0

    for i in range(f_frame, n_frame):
      motion = images[i][:, :, 1] / 255.
      motion_next = images[i + 1][:, :, 1] / 255.
      levels = 4
      k_size = 11
      k_type = "gaussian"
      sigma = k_size
      interpolation = cv2.INTER_AREA  # You may try different values
      border_mode = cv2.BORDER_REFLECT101  # You may try different values

      u, v = hierarchical_lk(motion, motion_next, levels, k_size,
                                 k_type, sigma, interpolation, border_mode)

      # absolute magnitude of displacement vector
      mag1 = np.sqrt(np.square(u) + np.square(v))

      # threshold vector that move less than 1 pixel
      mag2 = np.where(mag1 < 1, 0, mag1)

      # median of moving vetors
      median = np.median(np.where(mag2 > 0))

      #masking vector that is twice the median only - to filter negligible noisy movement
      mag3 = np.where(mag2 > 2 * median, 0, mag2)

      #accumulate moving pixel count, and displacement vector sum
      moving_pixel_sum += np.count_nonzero(mag3)
      displacement_sum += np.sum(mag3)

    movement_ratio = displacement_sum/moving_pixel_sum

    if movement_ratio >= 3.5: #highest displacement to moving pixel ratio
      video_class = 1 #running
    elif 2 <= movement_ratio < 3.5: #medium displacement to moving pixel ratio
      video_class = 2 #walking
    else: #lower displacement to moving pixel ratio
      video_class = 3 #clapping

    return video_class
