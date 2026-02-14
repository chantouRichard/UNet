import cv2
import os
import numpy as np
from scipy import ndimage
# import SimpleITK as sitk 

class vesselness2d:
    def __init__(self, image, sigma, spacing, tau):
        super(vesselness2d, self).__init__()
        # image 为numpy类型，表示n * m 的二维矩阵。
        self.image = image
        # sigma 为list 类型，表示高斯核的尺度。
        self.sigma = sigma
        # spacing 为list类型，表示.nii文件下某一切面下的体素的二维尺寸。如果输入图像本身为二维图像，则为[1,1],如果为三维图像，则为对应的space。
        self.spacing = spacing
        # tau 为float类型，表示比例系数。
        self.tau = tau
        # 图像尺寸
        self.size = image.shape
	
	# 使用特定的特定sigma尺寸下的高斯核对图像滤波
	# 这里作者并没有使用n*n的卷积核，而是分别使用n*1，1*n的卷积对图像进行x和y方向上的卷积，
	# 并且使用的是最原始的计算高斯函数得到卷积核，而不是直接用现成的高斯卷积核，
	# 通过证明可以发现在两方面的结果是等价的。
    def gaussian(self, image, sigma):

        siz = sigma * 6

        # ===== X方向 =====
        temp = round(siz / self.spacing[0] / 2)
        x = np.arange(-temp, temp + 1)

        H = np.exp(-(x ** 2) / (2 * (sigma / self.spacing[0]) ** 2))
        H = H / np.sum(H)

        # 关键修改：改成二维卷积核
        Hx = H.reshape(len(H), 1)   # (k,1)

        I = ndimage.convolve(image, Hx, mode='nearest')

        # ===== Y方向 =====
        temp = round(siz / self.spacing[1] / 2)
        x = np.arange(-temp, temp + 1)

        H = np.exp(-(x ** 2) / (2 * (sigma / self.spacing[1]) ** 2))
        H = H / np.sum(H)

        Hy = H.reshape(1, len(H))   # (1,k)

        I = ndimage.convolve(I, Hy, mode='nearest')

        return I

	# 求图像的梯度
    def gradient2(self, F, option):
        k = self.size[0]
        l = self.size[1]
        D = np.zeros(F.shape)
        if option == "x":
            D[0, :] = F[1, :] - F[0, :]
            D[k - 1, :] = F[k - 1, :] - F[k - 2, :]
            # take center differences on interior points
            D[1:k - 2, :] = (F[2:k - 1, :] - F[0:k - 3, :]) / 2
        else:
            D[:, 0] = F[:, 1] - F[:, 0]
            D[:, l - 1] = F[:, l - 1] - F[:, l - 2]
            D[:, 1:l - 2] = (F[:, 2:l - 1] - F[:, 0:l - 3]) / 2
        return D
        
	# 求海森矩阵中所需要的二阶偏导数
    def Hessian2d(self, image, sigma):
        image = self.gaussian(image, sigma)
        # image = ndimage.gaussian_filter(image, sigma, mode = 'nearest')
        Dy = self.gradient2(image, "y")
        Dyy = self.gradient2(Dy, "y")

        Dx = self.gradient2(image, "x")
        Dxx = self.gradient2(Dx, "x")
        Dxy = self.gradient2(Dx, 'y')
        return Dxx, Dyy, Dxy
	
	# 求解海森矩阵的两个特征值
	# 这里作者使用求根公式，将二阶海森矩阵展开，a=1,b=-(Ixx+Iyy),c=(Ixx*Iyy-Ixy*Ixy) 
	# 首先计算 sqrt(b^2 - 4ac),通过化简得到tmp
	# 最后得到两个特征值mu1，mu2，根据大小关系，大的为mu2，小的为mu1
    def eigvalOfhessian2d(self, Dxx, Dyy, Dxy):
        tmp = np.sqrt((Dxx - Dyy) ** 2 + 4 * (Dxy ** 2))
        # compute eigenvectors of J, v1 and v2
        mu1 = 0.5 * (Dxx + Dyy + tmp)
        mu2 = 0.5 * (Dxx + Dyy - tmp)
        # Sort eigen values by absolute value abs(Lambda1) < abs(Lambda2)
        indices = (np.absolute(mu1) > np.absolute(mu2))
        Lambda1 = mu1
        Lambda1[indices] = mu2[indices]

        Lambda2 = mu2
        Lambda2[indices] = mu1[indices]
        return Lambda1, Lambda2

    def imageEigenvalues(self, I, sigma):
        hxx, hyy, hxy = self.Hessian2d(I, sigma)
        c = sigma ** 2
        hxx = -c * hxx
        hyy = -c * hyy
        hxy = -c * hxy

        # 为了降低运算量，去掉噪声项的计算
        B1 = -(hxx + hyy)
        B2 = hxx * hyy - hxy ** 2
        T = np.ones(B1.shape)
        T[(B1 < 0)] = 0
        T[(B1 == 0) & (B2 == 0)] = 0
        T = T.flatten()
        indeces = np.where(T == 1)[0]

        hxx = hxx.flatten()
        hyy = hyy.flatten()
        hxy = hxy.flatten()

        hxx = hxx[indeces]
        hyy = hyy[indeces]
        hxy = hxy[indeces]


        lambda1i, lambda2i = self.eigvalOfhessian2d(hxx, hyy, hxy)
        lambda1 = np.zeros(self.size[0] * self.size[1], )
        lambda2 = np.zeros(self.size[0] * self.size[1], )

        lambda1[indeces] = lambda1i
        lambda2[indeces] = lambda2i

        # 去掉噪声
        lambda1[(np.isinf(lambda1))] = 0
        lambda2[(np.isinf(lambda2))] = 0

        lambda1[(np.absolute(lambda1) < 1e-4)] = 0
        lambda1 = lambda1.reshape(self.size)

        lambda2[(np.absolute(lambda2) < 1e-4)] = 0
        lambda2 = lambda2.reshape(self.size)
        return lambda1, lambda2
	
	# 血管强化
    def vesselness2d(self):
        print("调用vesselness2d")
        for j in range(len(self.sigma)):
            lambda1, lambda2 = self.imageEigenvalues(self.image, self.sigma[j])
            lambda3 = lambda2.copy()
            new_tau = self.tau * np.min(lambda3)
            lambda3[(lambda3 < 0) & (lambda3 >= new_tau)] = new_tau
            different = lambda3 - lambda2
            response = ((np.absolute(lambda2) ** 2) * np.absolute(different)) * 27 / (
                        (2 * np.absolute(lambda2) + np.absolute(different)) ** 3)
            response[(lambda2 < lambda3 / 2)] = 1
            response[(lambda2 >= 0)] = 0

            response[np.where(np.isinf(response))[0]] = 0
            if j == 0:
                vesselness = response
            else:
                vesselness = np.maximum(vesselness, response)
        vesselness[(vesselness < 1e-2)] = 0
        return vesselness
    
