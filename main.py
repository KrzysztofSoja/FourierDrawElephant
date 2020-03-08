import matplotlib.pyplot as plt
import numpy as np
import cv2
import imageio
import time
import torch
import torch.nn
import torch.optim
import torch.utils.data
import os

from tqdm import tqdm
from skimage.morphology import skeletonize
from typing import Tuple, List, Dict


def make_train_set(path: str) -> np.ndarray:
    """ Make array with points from pictures. """
    image = cv2.imread(path, 0)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    image = 255 - image
    image //= 255
    image = skeletonize(image)
    image = image.astype(np.uint8)

    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.squeeze(contours[0], axis=1)
    contours = contours - np.mean(contours)  # Normalize data.
    contours = contours / np.std(contours)
    return contours


class FourierDrawer:

    def __init__(self, data: np.ndarray, number_of_formulas: int, experiment_name: str = None):
        self.data = data
        self.number_of_formulas = number_of_formulas

        self.a = torch.rand((number_of_formulas, 2), requires_grad=True)
        self.b = torch.rand((number_of_formulas, 2), requires_grad=True)

        self.experiment_name = str(time.time()) if experiment_name is None else experiment_name
        FourierDrawer._make_dir_if_not_exists('./' + self.experiment_name)
        FourierDrawer._make_dir_if_not_exists('./' + self.experiment_name + '/images')
        FourierDrawer._make_dir_if_not_exists('./' + self.experiment_name + '/parameters')

    @staticmethod
    def _make_dir_if_not_exists(dir_name: str):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def _params(self) -> List[Dict[str, torch.Tensor]]:
        """ Return list of model parameters."""
        return [{'params': self.a}, {'params': self.b}]

    def _fourier_equitation(self, input: torch.Tensor) -> torch.Tensor:
        """Fourier equitation write as torch model."""
        k = torch.arange(start=0, end=self.number_of_formulas, out=torch.Tensor())

        sin = torch.sin(torch.mul(input, k))
        cos = torch.cos(torch.mul(input, k))
        sin = sin.repeat((1, 2)).view((-1, 2))
        cos = cos.repeat((1, 2)).view((-1, 2))

        psin = torch.mul(self.a, sin)
        pcos = torch.mul(self.b, cos)

        output = torch.add(psin, pcos)
        output = torch.sum(output, dim=(0,))

        return output

    def train(self, lr: float, steps: int, make_gif: bool=False):
        """Training model to imitate image."""
        optimizer = torch.optim.Adam(params=self._params(), lr=lr)
        loss = torch.nn.MSELoss()

        t = np.arange(0, 2*np.pi, 2*np.pi / len(self.data))
        acc_history = []

        for step in tqdm(range(steps), desc="Fitting equitation: "):
            acc = []
            for idx, _t in enumerate(t):
                output = self._fourier_equitation(torch.Tensor([_t]))
                output = loss(input=output, target=torch.Tensor(self.data[idx]))
                acc.append(output.item())
                output.backward()
                optimizer.step()
            self._save_parameters(str(step) + '.txt')

            acc_history.append(np.mean(acc))

            if make_gif:
                points = [self.calculate_point(_t) for _t in t]
                x = [point[0] for point in points]
                y = [point[1] for point in points]
                self._make_image(x, y, image_name=str(step))

        self._make_history_plot(acc_history)
        if make_gif:
            self._make_gif(steps)

    def calculate_point(self, t: float) -> Tuple[float, float]:
        """Calculate point from time parameter."""
        t = torch.Tensor([t])
        point = self._fourier_equitation(t)
        return point.detach().numpy()

    def _make_history_plot(self, history: List[float]):
        """Create and save plot of history learning."""
        plt.plot(history)
        path = self.experiment_name + '/acc_plot.png'
        plt.savefig(path)
        plt.close()

    def _make_image(self, x: List[float], y: List[float], image_name: str):
        plt.scatter(x, y)
        path = self.experiment_name + '/images/' + image_name + '.png'
        plt.savefig(path)
        plt.close()

    def _make_gif(self, image_number: int):
        """Create gif from images. Illustrate history of learning."""
        path_to_gif = self.experiment_name
        with imageio.get_writer(path_to_gif + '/move.gif', mode='I') as writer:
            for i in tqdm(range(image_number), desc='Creating gif: '):
                file_name = self.experiment_name + '/images/' + str(i) + '.png'
                image = imageio.imread(file_name)
                writer.append_data(image)

    def _save_parameters(self, file_name):
        """Save model parameters to txt file."""
        a = self.a.detach().numpy()
        b = self.b.detach().numpy()
        path = './' + self.experiment_name + '/parameters/' + file_name
        with open(path, 'w+') as file:
            file.write('a: ' + str(a) + '\n')
            file.write('b: ' + str(b) + '\n')

    def load_parameters(self, path):
        """Load parameters to model from txt file."""
        with open(path, 'r') as file:
            text = file.read()
            text = text[3:]

            a = ''
            for i in text:
                if i == 'b':
                    break
                a = a + i
                text = text[1:]
            self.a = np.array(a)
            b = text[3:]
            self.b = np.array(b)


if __name__ == "__main__":
    data = make_train_set('simple_dick.png')
    drawer = FourierDrawer(data, 4)
    drawer.train(lr=0.00005, steps=1000, make_gif=True)
