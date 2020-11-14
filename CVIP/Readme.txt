This directory includes the datasets as described in

E.Ardizzone, A. Bruno, G. Mazzola, "Copy-Move Forgery Detection by Matching Triangles of Keypoints", Information Forensics and Security, IEEE Transactions on, October 2015, Vol. 10, Issue 10 
pp. 2084-2094, DOI 10.1109/TIFS.2015.24457422015.

Please cite our work if you use our dataset. 

The dataset is composed as follows:

D0 includes 50 simply translated copies, and the related binary masks that indicate the source and the destination areas of the tampering.

D3 includes the original 50 not tampered images

D1-2 includes the tampered images in which every copy-pasted area is transformed according to the following transformations:

- rotation in the range of[-25°, 25°] with step 5°

- rotation in the range of [0°, 360°[ with a step of 30° 

- rotation in the range of [-5°, 5°] with a step of 1°

- scaling in the range of [0.25, 2] with step 0.25

- scaling in the range of[0.75, 1.25] with step 0.05

and the related binary masks that indicate the source and the destination areas of the tampering.

Each tampered image has been renamed according to the following rule
out_(r/s)(par)_im(num) where r/s is the type of transformation (rotation vs scaling), par is the parameter (angle of rotation or scaling factor), and num is the number of the image (the tampered images have been created starting from 20 original images).

in the D1-2 directory the tampered images are grouped into 20 sub-directories according to the original images. Each directory includes also the original image.



