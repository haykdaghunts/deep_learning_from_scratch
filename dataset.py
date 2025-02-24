import os
import numpy as np
from PIL import Image

class ImageFolderDataset:
    """Custom dataset class for image classification"""

    def __init__(self, root_path, transform=None, mode='train', split={'train': 0.6, 'val': 0.2, 'test': 0.2}, limit_files=None):
        self.root_path = root_path
        self.transform = transform
        self.split = split
        self.limit_files = limit_files  

        self.classes, self.class_to_idx = self._find_classes(self.root_path)
        all_images, all_labels = self.make_dataset(self.root_path, self.class_to_idx)
        

        self.images, self.labels = self.select_split(all_images, all_labels, mode)

    @staticmethod
    def _find_classes(directory):
        """Finds class folders and assigns labels."""
        classes = sorted([d.name for d in os.scandir(directory) if d.is_dir()])
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx

    @staticmethod
    def make_dataset(directory, class_to_idx):
        """Creates a dataset by scanning directories for images."""
        images, labels = [], []
        for target_class, label in class_to_idx.items():
            target_dir = os.path.join(directory, target_class)
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    if fname.endswith((".png", ".jpg", ".jpeg")):
                        images.append(os.path.join(root, fname))
                        labels.append(label)
        assert len(images) == len(labels)
        return images, labels

    def select_split(self, images, labels, mode):
        """Splits dataset into train, val, and test sets."""
        num_samples = len(images)
        num_train = int(num_samples * self.split['train'])
        num_valid = int(num_samples * self.split['val'])

        np.random.seed(0)
        rand_perm = np.random.permutation(num_samples)

        if mode == 'train':
            idx = rand_perm[:num_train]
        elif mode == 'val':
            idx = rand_perm[num_train:num_train + num_valid]
        elif mode == 'test':
            idx = rand_perm[num_train + num_valid:]
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from ['train', 'val', 'test'].")

        if self.limit_files is not None:  # checking before slicing
            idx = idx[:self.limit_files]

        return list(np.array(images)[idx]), list(np.array(labels)[idx])

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.images)

    @staticmethod
    def load_image_as_numpy(image_path):
        """Loads an image and converts it to a NumPy array."""
        return np.asarray(Image.open(image_path), dtype=np.float32)

    def __getitem__(self, index):
        """Returns a dictionary with image and label."""
        image = self.load_image_as_numpy(self.images[index])
        if self.transform:
            image = self.transform(image)  # applying transformations if provided
        return {"image": image, "label": self.labels[index]}






