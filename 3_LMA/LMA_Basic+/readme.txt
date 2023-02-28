Basic implementation of landmark morphing attacks (LMA).

Run generate_dataset.m to generate a dataset of face morphs

You will need to have [2] placed in the appropriate directory and may need to modify environment path variables.

To add \bin directory of [2] to windows PATH environment variable go to:
system properties -> advanced -> environment variables -> Path -> edit -> new

You will need to add [3] to the MATLAB path.
You can use MATLAB's add-on explorer to add [3] to the path.

Based on the combined morphing pipeline from [1].
Uses [2] for landmark location.
Uses [3] for image splicing.

[1] T. Neubert, A. Makrushin, M. Hildebrandt, C. Kraetzer and J. Dittmann, Extended StirTrace Benchmarking of Biometric and Forensic Qualities of Morphed Face Images, IET Biometrics, Volume 7, Issue 4, pp. 325–332, 2018
[2] https://github.com/YuvalNirkin/find_face_landmarks
[3] https://www.mathworks.com/matlabcentral/fileexchange/39438-fast-seamless-image-cloning-by-modified-poisson-equation




