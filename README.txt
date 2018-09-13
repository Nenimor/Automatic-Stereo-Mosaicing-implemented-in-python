Automatic “Stereo Mosaicing”, implemented in python (Image Processing course)

Input: A video of a scene, captured from left to right (due to camera rotation and/or translation).
Output: A short stereo panoramic video of the whole scene.

High-level of the algorithm:
The algorithm splits the video into a sequence of images, then performs the following process on each consecutive pair of images:
	• Finds the geometric transformation between the images by detecting Harris feature points, extracting their MOPS-like 			 descriptors, matching points’ descriptors by using the RANSAC algorithm.
After finding all transformations, the algorithm aligns all images into a specific surface, and stiches strips from them into a full panorama, using pyramid blending.
