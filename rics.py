import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


class Rics():

    def __init__(self, path_to_data, imsize=256, microscope='1-photon'):
        self.path = path_to_data
        self.directory = os.fsencode(path_to_data)
        self.imsize = imsize
        self.microscope = microscope

    def image_reader(self, img_path):
        '''
        Returns an array of image intensity
        :param img_path: path as string to image directory
        '''
        return cv2.imread(img_path, 0)

    def acf_single_image(self, img, img_path=False):
        if img_path != False:
            img = self.image_reader(img_path)
        power = np.real(np.fft.ifft2(np.fft.fft2(img) * np.conj(np.fft.fft2(img))))
        shift = np.fft.fftshift(power)
        normalize = shift / (np.mean(img)*np.mean(img)*len(img[0,:])*len(img[:,0])) - 1
        return normalize

    def generate_image_stack(self):
        image_stack = []
        for file in os.listdir(self.directory):
            filename = os.fsdecode(file)
            if filename.endswith('.tif') or filename.endswith('.tiff'):
                image_stack.append(self.image_reader(img_path=self.path + filename))
        return np.array(image_stack).astype(dtype=float)

    def acf_stack(self, frames_substract_moving_average=0, substract_average=False):
        image_stack = self.generate_image_stack()
        acf = []
        if substract_average or frames_substract_moving_average==1:
            for image in image_stack:
                image -= np.mean(image)
        elif frames_substract_moving_average > 1:
            for i in range(len(image_stack)-frames_substract_moving_average+1):
                image_sum = np.zeros(shape=np.array(image_stack[0]).shape)
                for j in range(frames_substract_moving_average):
                    image_sum += image_stack[i+j]
                image_stack[i] -= np.mean(image_sum)
        for image in image_stack:
            acf.append(self.acf_single_image(img=image))
        self.acf = np.mean(a=acf, axis=2)
        self.calc_acf = True
        return self.acf

    def rics_fitting(self):
        return None

    def roi_selection(self):
        return None

    def graph_single_acf(self, filename):
        acf = self.acf_single_image(img=None, img_path=self.path + filename)
        plt.figure('Single ACF for image: ' + filename)
        plt.title('ACF for single image: ' + filename)
        plt.imshow(acf, cmap='coolwarm')
        plt.show()

    def graph_acf(self):
        if not self.calc_acf:
            return None
        plt.figure('ACF - image stack')
        plt.imshow(self.acf, cmap='coolwarm')
        plt.title('ACF - Image Stack')
        plt.show()

if __name__ == '__main__':
    rod = Rics('/home/ferbellora/Documents/FCS/RICS_rhodamina_9/', microscope='1-photon')
    r = rod.acf_single_image(img=None, img_path=rod.path + '0.14 s.tif')
    rod.graph_single_acf('0.14 s.tif')

    rod2 = Rics('/home/ferbellora/Documents/FCS/RICS_rhodamina_9/', microscope='1-photon')
    rod2.acf_stack()
    rod2.graph_acf()

    rod3 = Rics('/home/ferbellora/Documents/FCS/RICS_rhodamina_9/', microscope='1-photon')
    rod3.acf_stack(substract_average=True)
    rod3.graph_acf()

    rod4 = Rics('/home/ferbellora/Documents/FCS/RICS_rhodamina_9/', microscope='1-photon')
    rod4.acf_stack(frames_substract_moving_average=5)
    rod4.graph_acf()
