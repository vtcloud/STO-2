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
server at The University of Arizona: http://soral.as.arizona.edu/STO2/


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
	




