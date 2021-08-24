from dataclasses import dataclass
from typing import Tuple
import numpy as np
import numpy.typing as npt
from .SPRM_pkg import *



class Features:

    def __init__(self, im: IMGstruct, mask, output_dir, options):
        self.output_directory = output_dir
        self.options = options
        self.superpixels = self.voxel_cluster(im, options)
        self.PCA = self.clusterchannels(im, mask)
        self.texture =

    def clusterchannels(self,
            im: IMGstruct, mask: MaskStruct ) -> np.ndarray:
        """
        cluster all channels using PCA
        """

        fname = im.get_name()
        inCells = mask.get_interior_cells()

        print("Dimensionality Reduction of image channels...")
        if self.options.get("debug"):
            print("Image dimensions before reduction: ", im.get_data().shape)

        pca_channels = PCA(n_components=self.options.get("num_channelPCA_components"), svd_solver="full")
        channvals = im.get_data()[0, 0, :, :, :, :]
        keepshape = channvals.shape
        channvals = channvals.reshape(
            channvals.shape[0], channvals.shape[1] * channvals.shape[2] * channvals.shape[3]
        )
        channvals = channvals.transpose()
        if self.options.get("debug"):
            print(channvals.shape)
        reducedim = pca_channels.fit_transform(channvals)
        if self.options.get("debug"):
            print("PCA Channels:", pca_channels, sep="\n")
            print("Explained variance ratio: ", pca_channels.explained_variance_ratio_)
            print("Singular values: ", pca_channels.singular_values_)
            print("Image dimensions before transposing & reshaping: ", reducedim.shape)
        reducedim = reducedim.transpose()
        reducedim = reducedim.reshape(reducedim.shape[0], keepshape[1], keepshape[2], keepshape[3])
        if self.options.get("debug"):
            print("Image dimensions after transposing and reshaping: ", reducedim.shape)

        # find pca comp and explained variance and concatenate
        a = pca_channels.explained_variance_ratio_
        b = abs(pca_channels.components_)
        c = np.column_stack((b, a))

        write_2_file(c, fname + "-channelPCA_summary", im, self.output_directory, inCells, self.options)

        return reducedim

    def voxel_cluster(self, im: IMGstruct) -> np.ndarray:
        """
        cluster multichannel image into superpixels
        """

        print("Clustering voxels into superpixels...")
        if im.get_data().shape[0] > 1:
            raise NotImplementedError("images with > 1 time point are not supported yet")

        channvals = im.get_data()[0, 0, :, :, :, :]
        keepshape = channvals.shape
        channvals = channvals.reshape(
            channvals.shape[0], channvals.shape[1] * channvals.shape[2] * channvals.shape[3]
        )
        channvals = channvals.transpose()
        # for some reason, voxel values are occasionally NaNs (especially edge rows)
        channvals[np.where(np.isnan(channvals))] = 0

        if self.options.get("zscore_norm"):
            channvals = stats.zscore(channvals)

        if self.options.get("debug"):
            print("Multichannel dimensions: ", channvals.shape)
        # get random sampling of pixels in 2d array
        np.random.seed(0)
        sampling = float(self.options.get("precluster_sampling"))
        samples = math.ceil(sampling * channvals.shape[0])
        # lower bound threshold on pixels
        if samples < self.options.get("precluster_threshold"):
            samples = self.options.get("precluster_threshold")
        idx = np.random.choice(channvals.shape[0], samples)
        channvals_random = channvals[idx]
        # kmeans clustering of random sampling
        print("Clustering random sample of voxels...")
        stime = time.monotonic() if self.options.get("debug") else None

        voxelbycluster = KMeans(n_clusters=self.options.get("num_voxelclusters"), random_state=0).fit(
            channvals_random
        )

        if self.options.get("debug"):
            print("random sample voxel cluster runtime: " + str(time.monotonic() - stime))
        cluster_centers = voxelbycluster.cluster_centers_
        # fast kmeans clustering with inital centers
        print("Clustering voxels with initialized centers...")
        stime = time.monotonic() if self.options.get("debug") else None
        voxelbycluster = KMeans(
            n_clusters=self.options.get("num_voxelclusters"),
            init=cluster_centers,
            random_state=0,
            max_iter=100,
            verbose=0,
            n_init=1,
        ).fit(channvals)

        if self.options.get("debug"):
            print("Voxel cluster runtime: ", time.monotonic() - stime)
        # returns a vector of len number of voxels and the vals are the cluster numbers
        voxelbycluster_labels = voxelbycluster.labels_
        voxelbycluster_labels = voxelbycluster_labels.reshape(keepshape[1], keepshape[2], keepshape[3])

        if self.options.get("debug"):
            print("Cluster Label dimensions: ", voxelbycluster_labels.shape)
            print("Number of unique labels:")
            print(len(np.unique(voxelbycluster_labels)))

        return voxelbycluster_labels

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
    feature: np.ndarray

