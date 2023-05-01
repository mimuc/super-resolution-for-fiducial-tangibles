# Deep Learning Super-Resolution Network Facilitating Fiducial Tangibles on Capacitive Touchscreens
Repository of the CHI'23 paper "Deep Learning Super-Resolution Network Facilitating Fiducial Tangibles on Capacitive Touchscreens"

Over the last few years, we have seen many approaches using tangibles to address the limited expressiveness of touchscreens. Mainstream tangible detection uses fiducial markers embedded in the tangibles. However, the coarse sensor size of capacitive touchscreens makes tangibles bulky, limiting their usefulness. We propose a novel deep-learning super-resolution network to facilitate fiducial tangibles on capacitive touchscreens better. In detail, our network super-resolves the markers enabling off-the-shelf detection algorithms to track tangibles reliably. Our network generalizes to unseen marker sets, such as AprilTag, ArUco, and ARToolKit. Therefore, we are not limited to a fixed number of distinguishable objects and do not require data collection and network training for new fiducial markers. With extensive evaluation, including real-world users and five showcases, we demonstrate the applicability of our open-source approach on commodity mobile devices and further highlight the potential of tangibles on capacitive touchscreens.

## Citing our work
This work can be cited as follows:
<pre>
@inproceedings{rusu2023deep,
title = {Deep Learning Super-Resolution Network Facilitating Fiducial Tangibles on Capacitive Touchscreens},
author = {Rusu, Marius and Mayer, Sven},
year = {2023},
booktitle = {Proceedings of the 42st ACM Conference on Human Factors in Computing Systems},
address = {Hamburg, Germany},
serires = {CHI'23},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi={10.1145/3544548.3580987}
}
</pre>

--- 

The Python notebook `notebook` is well documented and includes preprocessing, network training and evaluation.
The folder `Data` contains raw capacitive and OptiTrack data. The subfolder `Processed` contains processed datafiles. The filenames indicate the fiducial marker ID and the condition e.g.:

*   `id1_4_8` -> AprilTag 36h11 marker with ID 1 and condition SMALL (4mm -> 8mm)
*   `id2t_6_12` -> AprilTag 15h6 marker with ID 2 and condition LARGE (6mm -> 12mm)
*   `id3a_4_8` -> ArUco original marker with ID 3 and condition SMALL (4mm -> 8mm)
*   `id4k_6_12` -> ArtoolKit 4x4 marker with ID 4 and condition LARGE (6mm -> 12mm)

The folder `Models` contains the networks featured in the thesis.
The folders `Logs`, `Eval` and `Figs` do not contain content from the thesis.
They only contain test content that can be generated with the notebook e.g. logs of test network training.

The folder `artk` contains the JavaScript/Python evaluation for the ARToolKit 4x4 markers.
For this, open `python_server_8080.py` and edit the dataset and log file.
Now run `python_server_8080.py`, then run `browser_8080.py`. 
Alternatively, open `index.html` in a browser manually to view the detection in real-time.
The results are written to the specified log file and can be evaluated with `eval.py`.
