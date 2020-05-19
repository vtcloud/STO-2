# STO-2 Data Reduction - Method 2

Python scripts related to Submillimeter Telescope Observatory-2
(STO-2) data reduction.

Thise repository, https://github.com/vtcloud/STO-2, provides the notebooks used for an alternate
reduction of the STO-2 data of the Eta Carinae 2 region. This
alternate data reduction was performed in order to confirm the results
of the primary data reduction using machine learning (see:
https://github.com/seoym3919/STO2_PIPELINE ). The second method for
the data reduction was required to confirm the results of the primary
data reduction pipeline because of the severity of instrumental effects impacting
the spectra. E.g., some spectra show fringe that vary across the
spectrum and have amplitude of several times the source signal. These
instrument effects had to be carefully removed while keeping the
desired measure signal unchanged.

The full data set for the reduction can be downloaded from the STO-2
server at The University of Arizona:

http://soral.as.arizona.edu/mediawiki/index.php?title=STO-2_Data_Releases_%26_Docs#Obtaining_Level_0.7_Products

The user should download Level 0.7 data since these data include
essential modifications of the FITS header. The scripts provided here
have only been used to reduce the data in directories 03800
to 03995. These directories are also included in the pre-bundled data
set at http://soral.as.arizona.edu/STO2/level0.7/bundles/etaCar2-spi_3751-4124.txz. As noted below,
the data should be placed into the directory "./Data/level0.7" relative to the 
directory from which the notebooks will be executed (or modify
directory entries in the notebooks appropriately for any other configuration).


The scripts include (in approximate order of execution):

    EtaCar2_despike_OTF_3801_L0.7_2Steps.ipynb
    Eta_Car_despike_mask_checking.ipynb

    EtaCar2_Coverage_Map_OTF_0.7_v1.ipynb

    EtaCar2_OTF_L0.7-L1.0_processing_v4_20200326.ipynb
    EtaCar2_OTF_L0.7-L1.0_processing_v4.ipynb

    EtaCar2_plot_OTF_page_per_scan_v4a_w_YS_spec.ipynb

    EtaCar2_plot_full_spectral_map_CII.ipynb


The following python functions and files are used by the scripts (and are provided):

    ALSFitter.py
    jupyter_custom.css
    STO2_v35.py
	
Python packages required (use the latest version available!):

    _cffi_backend
    aplpy
    astropy
    datetime
    glob
    inspect
    IPython
    matplotlib
    ntpath
    numpy
    os
    pandas
    scipy
    sys
    time
    warnings



To run the notebooks, the following directories are required:

    ./Data
    ./Data/DSS
    ./Data/level0.7            # this directory contains the data from the STO-2 data server
    ./Data/processed
    ./Data/Ref_Specs




