This repository contains the code and data for the paper "Deep Learning Super-Resolution Network Facilitating Fiducial Tangibles on Capacitive Touchscreens".
PLEASE NOTE: The easy-to-use pickled data cannot be shared here due to the filesize limitation. Instead, the processing must be repeated.

The Python notebook *notebook* is well documented and includes preprocessing, network training and evaluation.
The folder *Data* contains raw capacitive and OptiTrack data. The subfolder *Processed* contains processed datafiles. The filenames indicate the fiducial marker ID and the condition e.g.:

*   id1_4_8 -> AprilTag 36h11 marker with ID 1 and condition SMALL (4mm -> 8mm)
*   id2t_6_12 -> AprilTag 15h6 marker with ID 2 and condition LARGE (6mm -> 12mm)
*   id3a_4_8 -> ArUco original marker with ID 3 and condition SMALL (4mm -> 8mm)
*   id4k_6_12 -> ArtoolKit 4x4 marker with ID 4 and condition LARGE (6mm -> 12mm)

The folder *Models* contains the networks featured in the thesis.
The folders *Logs*, *Eval* and *Figs* do not contain content from the thesis.
They only contain test content that can be generated with the notebook e.g. logs of test network training.

The folder *artk* contains the JavaScript/Python evaluation for the ARToolKit 4x4 markers.
For this, open *python_server_8080.py* and edit the dataset and log file.
Now run *python_server_8080.py*, then run *browser_8080.py*. 
Alternatively, open *index.html* in a browser manually to view the detection in real-time.
The results are written to the specified log file and can be evaluated with *eval.py*
