from pathlib import Path
from typing import Dict

import numpy as np
from aicsimageio import AICSImage


class IMGstruct:
    """
    Main Struct for IMG information
    """

    img: AICSImage
    data: np.ndarray
    path: Path
    name: str

    def __init__(self, path: Path, options):
        self.img = self.read_img(path, options)
        self.data = self.read_data(options)
        self.path = path
        self.name = path.name
        self.channel_labels = self.read_channel_names()

    def set_img(self, img):
        self.img = img

    def get_meta(self):
        return self.img.metadata

    # def quit(self):
    #     self.img = None
    #     self.data = None
    #     self.path = None
    #     self.name = None
    #     self.channel_labels = None

    @staticmethod
    def read_img(path: Path, options: Dict) -> AICSImage:
        img = AICSImage(path)
        if not img.metadata:
            print("Metadata not found in input image")
            # might be a case-by-basis
            # img = AICSImage(path), known_dims="CYX")

        return img

    def read_data(self, options):
        data = self.img.data
        dims = data.shape

        # Haoran: hot fix for 5 dims 3D IMC images
        # if len(self.img.data.shape) == 5:
        #     data = self.img.data[:, :, :, :, :]
        #     dims = data.shape
        #     s, c, z, y, x = dims[0], dims[1], dims[2], dims[3], dims[4]
        # else:
        #     data = self.img.data[:, :, :, :, :, :]
        #     dims = data.shape
        #     s, t, c, z, y, x = dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]
        #     if t > 1:
        #         data = data.reshape((s, 1, t * c, z, y, x))

        # older version of aicsimageio<3.2
        if len(dims) == 6:
            s, t, c, z, y, x = dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]
            if t > 1:
                data = data.reshape((s, 1, t * c, z, y, x))

        # newer version of aicsimageio>4.0 -> correct dimensions
        elif len(dims) == 5:
            data = data[np.newaxis, ...]

        else:
            print(
                "image/expressions dimensions are incompatible. Please check that its in correct format."
            )

        # convert data to a float for normalization downstream
        data = data.astype("float")
        assert data.dtype == "float"

        return data

    def read_channel_names(self):
        img: AICSImage = self.img
        # cn = get_channel_names(img)
        cn = img.channel_names
        print("Channel names:")
        print(cn)

        return cn

    def get_name(self):
        return self.name

    def get_channel_labels(self):
        return self.channel_labels

    def set_channel_labels(self, channel_list):
        self.channel_labels = channel_list


class MaskStruct(IMGstruct):
    """
    Main structure for segmentation information
    """

    all_cells: set[int]
    interior_cells: set[int]
    bad_cells: set[int]

    def __init__(self, path: Path, options):
        super().__init__(path, options)
        self.bestz = self.get_bestz()
        self.bad_cells = set()
        self.ROI = []
        self.find_edge_cells()

    def read_channel_names(self):
        img: AICSImage = self.img
        # cn = get_channel_names(img)
        cn = img.channel_names
        print("Channel names:")
        print(cn)

        # hot fix to channel names expected
        expected_names = ["cell", "nuclei", "cell_boundaries", "nucleus_boundaries"]

        for i in range(len(cn)):
            cn[i] = expected_names[i]

        return cn

    def get_labels(self, label):
        return self.channel_labels.index(label)

    def set_bestz(self, z):
        self.bestz = z

    def get_bestz(self):
        return self.bestz

    def read_data(self, options):
        bestz = []
        data = self.img.data
        dims = data.shape
        # s,t,c,z,y,x = dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]

        # aicsimageio > 4.0
        if len(dims) == 5:
            data = data[np.newaxis, ...]

        check = data[:, :, :, 0, :, :]
        check_sum = np.sum(check)
        if (
            check_sum == 0 and options.get("image_dimensions") == "2D"
        ):  # assumes the best z is not the first slice
            print("Duplicating best z to all z dimensions...")
            for i in range(0, data.shape[3]):
                x = data[:, :, :, i, :, :]
                y = np.sum(x)
                #                print(x)
                #                print(y)
                if y > 0:
                    bestz.append(i)
                    break
                else:
                    continue

            if options.get("debug"):
                print("Best z dimension found: ", bestz)
            # data now contains just the submatrix that has nonzero values
            # add back the z dimension
            data = x[:, :, :, np.newaxis, :, :]
            # print(data.shape)
            # and replicate it
            data = np.repeat(data, dims[3], axis=3)
            # print(data.shape)
            # set bestz
        else:  # 3D case or that slice 0 is best
            if dims[2] > 1:
                bestz.append(int(data.shape[3] / 2))
            else:
                bestz.append(0)

        # set bestz
        self.set_bestz(bestz)

        # check to make sure is int64
        data = data.astype(int)
        assert data.dtype == "int"

        return data

    @property
    def usable_cells(self) -> set[int]:
        return self.interior_cells - self.bad_cells

    @property
    def cell_index(self) -> list[int]:
        return sorted(self.usable_cells)

    @property
    def cell_count(self) -> int:
        return len(self.usable_cells)

    def get_cell_selection_vector(self) -> np.ndarray:
        """
        Returns a boolean vector for selecting usable cells (interior,
        didn't fail outline PCA computation) in a sorted vector or list
        of all distinct pixel values in the mask (including 0 for
        background pixels not in any cell).
        """
        return np.array([idx in self.usable_cells for idx in sorted(self.all_cells | {0})])

    def add_bad_cell(self, cell_index):
        self.bad_cells.add(cell_index)

    def add_bad_cells(self, cell_indexes):
        self.bad_cells.update(cell_indexes)

    def set_ROI(self, ROI):
        self.ROI = ROI

    def get_ROI(self):
        return self.ROI

    def find_edge_cells(self):
        z = self.data.shape[3]
        channels = self.data.shape[2]
        data = self.data[0, 0, 0, :, :, :]
        border = set()

        if z > 1:  # 3D case
            for i in range(0, z):
                border.update(data[i, 0, :-1])  # Top row (left to right), not the last element.
                border.update(
                    data[i, :-1, -1]
                )  # Right column (top to bottom), not the last element.
                border.update(
                    data[i, -1, :0:-1]
                )  # Bottom row (right to left), not the last element.
                border.update(
                    data[i, ::-1, 0]
                )  # Left column (bottom to top), all elements element.

        else:  # 2D case
            bestz = self.get_bestz()[0]
            data = self.data[0, 0, :, bestz, :, :]

            for i in range(0, channels):
                border.update(data[i, 0, :-1])  # Top row (left to right), not the last element.
                border.update(
                    data[i, :-1, -1]
                )  # Right column (top to bottom), not the last element.
                border.update(
                    data[i, -1, :0:-1]
                )  # Bottom row (right to left), not the last element.
                border.update(
                    data[i, ::-1, 0]
                )  # Left column (bottom to top), all elements element.

        self.edge_cells = border - {0}

        mask_data = self.data[0, 0, 0, :, :, :]
        self.all_cells = set(mask_data.flat) - {0}
        self.interior_cells = self.all_cells - border
