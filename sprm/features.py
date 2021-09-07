from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import numpy.typing as npt
from .SPRM_pkg import *
from .outlinePCA import getparametricoutline, getcellshapefeatures


class Features:
    """
    

    """

    def __init__(self, im: IMGstruct, mask, output_dir, options):
        self.output_directory = output_dir
        self.options = options

        self.superpixels = self.voxel_cluster(im)
        self.PCA = self.cluster_channels(im, mask)

        texture, texture_headers = glcmProcedure(im, mask)
        self.texture = Feature(im.get_name + '_texture', texture.shape, texture_headers, texture)

        outline_vectors, cell_polygons = getparametricoutline(mask, )

    def cluster_channels(self,
                         im: IMGstruct, mask: MaskStruct) -> np.ndarray:
        """
        cluster all channels using PCA
        """

        filename = im.get_name()
        inCells = mask.get_interior_cells()

        print("Dimensionality Reduction of image channels...")
        if self.options.get("debug"):
            print("Image dimensions before reduction: ", im.get_data().shape)

        pca_channels = PCA(n_components=self.options.get("num_channelPCA_components"), svd_solver="full")
        channel_data = im.get_data()[0, 0, :, :, :, :]
        keep_shape = channel_data.shape
        channel_data = channel_data.reshape(
            channel_data.shape[0], channel_data.shape[1] * channel_data.shape[2] * channel_data.shape[3]
        )
        channel_data = channel_data.transpose()
        if self.options.get("debug"):
            print(channel_data.shape)
        reduced_image = pca_channels.fit_transform(channel_data)
        if self.options.get("debug"):
            print("PCA Channels:", pca_channels, sep="\n")
            print("Explained variance ratio: ", pca_channels.explained_variance_ratio_)
            print("Singular values: ", pca_channels.singular_values_)
            print("Image dimensions before transposing & reshaping: ", reduced_image.shape)
        reduced_image = reduced_image.transpose()
        reduced_image = reduced_image.reshape(reduced_image.shape[0], keep_shape[1], keep_shape[2], keep_shape[3])
        if self.options.get("debug"):
            print("Image dimensions after transposing and reshaping: ", reduced_image.shape)

        # find pca comp and explained variance and concatenate
        a = pca_channels.explained_variance_ratio_
        b = abs(pca_channels.components_)
        c = np.column_stack((b, a))

        write_2_file(c, filename + "-channelPCA_summary", im, self.output_directory, inCells, self.options)

        return reduced_image

    def voxel_cluster(self, im: IMGstruct) -> np.ndarray:
        """
        cluster multichannel image into superpixels
        """

        print("Clustering voxels into superpixels...")
        if im.get_data().shape[0] > 1:
            raise NotImplementedError("images with > 1 time point are not supported yet")

        channel_data = im.get_data()[0, 0, :, :, :, :]
        keep_shape = channel_data.shape
        channel_data = channel_data.reshape(
            channel_data.shape[0], channel_data.shape[1] * channel_data.shape[2] * channel_data.shape[3]
        )
        channel_data = channel_data.transpose()
        # for some reason, voxel values are occasionally NaNs (especially edge rows)
        channel_data[np.where(np.isnan(channel_data))] = 0

        if self.options.get("z-score_norm"):
            channel_data = stats.zscore(channel_data)

        if self.options.get("debug"):
            print("Multichannel dimensions: ", channel_data.shape)
        # get random sampling of pixels in 2d array
        np.random.seed(0)
        sampling = float(self.options.get("pre-cluster_sampling"))
        samples = math.ceil(sampling * channel_data.shape[0])
        # lower bound threshold on pixels
        if samples < self.options.get("pre-cluster_threshold"):
            samples = self.options.get("pre-cluster_threshold")
        idx = np.random.choice(channel_data.shape[0], samples)
        channel_data_random = channel_data[idx]
        # k-means clustering of random sampling
        print("Clustering random sample of voxels...")
        start_time = time.monotonic() if self.options.get("debug") else None

        voxel_by_cluster = KMeans(n_clusters=self.options.get("Number of Voxel Clusters"), random_state=0).fit(
            channel_data_random
        )

        if self.options.get("debug"):
            print("random sample voxel cluster runtime: " + str(time.monotonic() - start_time))
        cluster_centers = voxel_by_cluster.cluster_centers_
        # fast kmeans clustering with inital centers
        print("Clustering voxels with initialized centers...")
        start_time = time.monotonic() if self.options.get("debug") else None
        voxel_by_cluster = KMeans(
            n_clusters=self.options.get("num_voxelclusters"),
            init=cluster_centers,
            random_state=0,
            max_iter=100,
            verbose=0,
            n_init=1,
        ).fit(channel_data)

        if self.options.get("debug"):
            print("Voxel cluster runtime: ", time.monotonic() - start_time)
        # returns a vector of len number of voxels and the vals are the cluster numbers
        voxelbycluster_labels = voxel_by_cluster.labels_
        voxelbycluster_labels = voxelbycluster_labels.reshape(keep_shape[1], keep_shape[2], keep_shape[3])

        if self.options.get("debug"):
            print("Cluster Label dimensions: ", voxelbycluster_labels.shape)
            print("Number of unique labels:")
            print(len(np.unique(voxelbycluster_labels)))

        return voxelbycluster_labels

    def glcmProcedure(self, im: IMGstruct, mask: MaskStruct):

        """
        Wrapper for GLCM
        """

        print("GLCM calculation initiated")

        angle = self.options.get("glcm_angles")
        distances = self.options.get("glcm_distances")
        angle = "".join(angle)[1:-1].split(",")
        distances = "".join(distances)[1:-1].split(",")
        angle = [int(i) for i in angle][0]  # Only supports 0 for now
        distances = [int(i) for i in distances]
        stime = time.monotonic()
        texture, texture_names = self.__glcm(
            im, mask, angle, distances)
        print("GLCM calculations completed: " + str(time.monotonic() - stime))

        return texture, texture_names

    def __glcm(self,
               im,
               mask,
               angle,
               distances,
               ):
        """
        By: Young Je Lee and Ted Zhang
        """

        filename = im.get_name()
        ROI_coords = mask.get_ROI()
        cellidx = mask.get_cell_index()

        colIndex = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
        inCells = mask.get_interior_cells().copy()
        texture_all = np.zeros(
            (2, len(inCells), im.get_data().shape[2], len(colIndex) * len(distances))
        )

        # get headers
        channel_labels = im.get_channel_labels().copy() * len(distances) * len(colIndex) * 2
        maskn = (
                mask.get_channel_labels()[0:2] * im.get_data().shape[2] * len(distances) * len(colIndex)
        )
        maskn.sort()
        cols = colIndex * im.get_data().shape[2] * len(distances) * 2
        dlist = distances * im.get_data().shape[2] * len(colIndex) * 2
        column_format = ["{channeln}_{mask}: {col}-{d}" for i in range(len(channel_labels))]
        header = list(
            map(
                lambda x, y, z, c, t: x.format(channeln=y, mask=z, col=c, d=t),
                column_format,
                channel_labels,
                maskn,
                cols,
                dlist,
            )
        )

        for i in range(2):
            for idx in range(len(inCells)):  # For each cell
                curROI = ROI_coords[i][inCells[idx]]

                if curROI.size == 0:
                    continue

                xmax, xmin, ymax, ymin = (
                    np.max(curROI[1]),
                    np.min(curROI[1]),
                    np.max(curROI[0]),
                    np.min(curROI[0]),
                )

                imga = im.get_data()

                for j in range(len(im.channel_labels)):  # For each channel

                    # convert to uint
                    imgroi = imga[0, 0, j, 0, curROI[0], curROI[1]]
                    index = np.arange(imgroi.shape[0])

                    # make cropped 2D image
                    img = np.zeros((xmax - xmin + 1, ymax - ymin + 1))
                    xn = curROI[1] - xmin
                    yn = curROI[0] - ymin
                    img[xn, yn] = imgroi[index]
                    img = (img / img.max()) * 255
                    img = img.astype(np.uint8)

                    for d in range(len(distances)):
                        result = greycomatrix(
                            img, [distances[d]], [angle], levels=256
                        )  # Calculate GLCM
                        result = result[
                                 1:, 1:
                                 ]  # Remove background influence by delete first row & column

                        for ls in range(len(colIndex)):  # Get properties
                            texture_all[i, idx, j, d + ls] = greycoprops(
                                result, colIndex[ls]
                            ).flatten()[0]

        ctexture = np.concatenate(texture_all, axis=1)
        ctexture = ctexture.reshape(len(inCells), -1)

        # For csv writing
        write_2_csv(header, ctexture, filename + "_" + "texture", self.output_dir, cellidx, self.options)

        # add timepoint dim so that feature is in sync
        texture_all = texture_all[np.newaxis, :, :, :, :]

        return texture_all, header


@dataclass
class Feature:
    """
        Default feature class for SPRM

        Features:
            Covariance Matrix
            Mean Vectors
            Total Intensity
            Shape Vectors
            Texture
            tSNE
    """

    name: str
    size: Tuple[int, ...]
    column_headers: List[str, ...]
    feature: np.ndarray

    def __post_init__(self):
