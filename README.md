# **Rubik's Cube CV: Machine Vision Project**

Wynona Brinkmann, Darian Jimenez, and Rohan Bendapudi

For Olin ENGR3590 Computational Introduction to Robotics

# **Project Overview**

For our machine vision project, we designed a Rubik’s Cube detection solution that captures each face of the Rubik's Cube and digitally reconstructs it. In our solution, the user will bring each face of the Rubik’s Cube parallel to the camera, and our (stretch) solution will auto-capture the face using a trained YOLOE model (an image segmentation model), identify the colors, and save the face to a dictionary. After scanning each face, our software will compute the state of the Rubik’s Cube in cubestring notation and provide the cubestring state to the Kociemba solver – which is an external Rubik’s Cube solver library. Once we receive the solved state, we utilize the twistyplayer library to visualize the steps to solve the Rubik’s Cube. 

# **Overall System Architecture**

![][image1]

For our Rubik’s Cube CV project, we created an MVP solution and a stretch goal solution. In our MVP solution, the user lines up the Rubik’s Cube face to a 3x3 grid on the screen and manually saves the state of the Rubik’s Cube face. In our stretch goal solution, the user displays the Rubik’s Cube face to the camera and a trained YOLOE segmentation model \`yoloe-26n-seg\` will segment the Rubik’s Cube face and pixel position in the captured video. Then, we isolate the segmented Rubik’s Cube image into a new image and line up a 3x3 grid with the isolated Rubik’s Cube. 

For the MVP solution, we created a UI which instructs a user on how to scan the cube to generate a 3D solver which they can then follow. When scanning each face of the cube, the user has to line it up with a 3x3 grid on the screen, and once the user is happy with their alignment, they can press the space bar to save. We also show them the preliminary HSV colors of each tile in the user interface so they can adjust their lighting/environment if the cube colors are way off (although these get adjusted later in post processing to be more accurate anyway). 

In the stretch goal solution, when we scan each face (again with a certain orientation, but this time easier to work with). In the stretch goal solution, we also expect that the user follows a specific rotation pattern to scan the cube (Right, Right, Up, Right, Right).

Then, after scanning each face, we compute the cubestring by mapping the color of the center tile of each face to the Up, Down, Right, Left, Front, and Back directions. In our MVP this is given as instructions on a screen, but with the auto scanning, we ensure the user only has to rotate the cube in one direction (x y z), scanning in the order (orange, green, red, white, blue, yellow). Then, we assign all corresponding colors on the Rubik’s cube to these directions in cube notation, creating a cubestring. 

In both solutions, after computing the cubestring, we pass the cubestring to the Kociemba solver, which calculates the moves in cube notation to solve the Rubik’s Cube. Finally, we use the twistyplayer library to visualize the moves to solve the Rubik’s Cube with a 3D model a user can easily. 

# **In-Depth Iterative Implementation Design**

## **Face Scanning**

In our function read\_face() we look to read the colors of the 9 squares on the cube. To do so the og\_image is loaded and a previous function face\_box() is called which provides the region of interest by providing the bounding box. To decouple the lighting variations from the original BGR (Blue, Green, Red) image we converted it to two color spaces: HSV (Hue, Saturation, Value) for hard-coded classification values of color and LAB (lightness, green-red axis, and blue-yellow axis) for downstream calibration and because it provided more information. Because of the addition of a bounding box it could easily be partitioned into 9 equal cells where within each square a region of interest (id\_radius) is sampled. Computing the np.median across each of these helps suppress the camera sensor noise and specular highlights from the rubik cube image.   
![][image2]  
*Fig 1\. This shows an early attempt at the bounding box with an overlay of color identification. By including this output at the end of the function it provided a visual check for how the image was being processed and the handling of our algorithm.*

## **(MVP) Cube Scanning**

Next, for our initial MVP solution, we implemented a manual scanning process. First the user lines up each face of the cube to a grid we display on the UI. We give the user a specific sequence of instructions to scan each face and once the user lines up each face with the grid (with proper orientation) they press the spacebar to save it. To improve upon our initial HSV implementation, this saves 9 LAB values we return with our read\_face function. We still use the HSV values and present them to the user in the initial scanning to ensure the user has some idea of the colors they are scanning. In the case the user does something incorrectly, we have controls to go back to the last face or quit scanning. 

To improve color classification (our main issue which we touch on later), we used the center piece of each face as the reference color (since the center piece of each face of the Rubik's Cube determines what color the side is, we know what color it is). We store the LAB values for each center piece in a reference dictionary, and the 9 sampled colors in a separate dictionary. After we scan all the faces, we compare the 54 pieces (6 sides \* 9 pieces per side), using weighted euclidean distance in LAB space. The closest reference center determines the color of each piece. This calibration was necessary since subtle changes in lighting greatly affected our scans and during testing we would often end up with scans that were impossible to to create using the turns we could do on a Rubik's Cube, and thus were impossible to solve 

![][image3]  
*Fig 2\. UI of our initial MVP cube scanner with grid and HSV Values*

## **Kociemba Integration**

After scanning all 6 faces, we reconstruct the Rubik's Cube using the colors we detect. First we create a dictionary mapping each color to its corresponding face in cube notation. For example, the white face is always held on top, so we use the notation U (Up) to describe it. Green is in the front so we use F (Front), etc. We then iterate through the scanned faces and convert each piece's classified color into its corresponding letter, which in the end generates a 54 character string. We also check that each color appears 9 times each to make sure the cube is actually potentially solvable, if not we raise an error and make the user rescan. 

We then pass the string to kociemba.solve(), which determines the sequence of moves needed to solve the cube, again, in cube notation (U R F L B D). This then goes to the TwistyPlayer library to visualize the solution on a 3D Rubik’s cube. To show the initial scramble we reverse the solution given to us by kociemba, scramble the cube, then the user can press play and follow the 3D cube as the scramble is undone.

## **(Stretch Goal) YOLOE Autodetection**

One key design decision was to build upon our manual scanning method with a YOLOE autodetection method. Our manual scanning method required the user to line up the Rubik’s Cube in a green grid and press the spacebar to take a photo of the Rubik’s Cube face. We felt that lining up the cube to the grid and pressing the spacebar would be difficult to coordinate – and slowed down the Rubik’s Cube reconstruction process. 

After some investigation, we learned about YOLOE (You Only Look Once \- Everything) for image segmentation, which we decided to use in our stretch goal solution. After some challenges with YOLOE (highlighted in Challenges and Learnings), we trained a custom, nano-sized YOLOE model on a dataset of Rubik’s Cube faces. Using this trained YOLOE model, we segment out Rubik’s Cube faces with a detection confidence score higher than 0.87 and isolate a new image, centered with the segmented Rubik’s Cube face. From here, we line up the Rubik’s Cube with a 3x3 (similar to the manual scanning process) and store each tile’s LAB space color values (using HSV values initially to show the user preliminary values). As mentioned in the System Architecture, the user has to use a certain order, but it requires less movement, is more intuitive to follow, and prevents duplicate face scans. 

# **Challenges and Learnings**

## **Edge Detection**

One challenge encountered early on in the project, that had to be mitigated at each iteration of the project, was noise. In figure 2 a rubik’s cube can be seen after it was grayscale and edge detection applied. What can also be seen are background objects outlined. When initially attempting to use the completed edges of the cube as the different points of classification this background noise created issues as too many edges were detected. To mitigate this logic a proper square was used as the rubik’s cube will always have three columns and three rows.   
![][image4]  
*Fig 3\. Rubik’s cube after grayscale and Canny edge detection applied.*

## **Color Classification** 

By far our biggest issue was color classification. We initially started off with hardcoded HSV values, however, these weren’t always accurate especially when changing cubes or having different lighting conditions. Oftentimes blue would get confused with white due to reflections giving it a lower saturation. The same issue would happen with red/orange and yellow/green because of how close they are in value. 

To fix this we implemented LAB color space instead of HSV, and since LAB separates lightness from color, we ended up getting a bit better accuracy. There was still the issue of using predefined values, and the shades change depending on what cube you use and the lighting of the room. To overcome this, we took a sample from each center piece of each side (since each center determines the color of the face of the cube, we know what it is), and compared those LAB values to the 54 other tiles. As aforementioned, we then use weighted euclidean values to determine the actual colors. We gave less emphasis to the lightness, and more to the color channels, giving less emphasis to lighting conditions. This greatly improved accuracy and since we haven’t had as many issues with color detection. 

To further improve upon this we attempted to use the skimage library with built in library `skimage.color.deltaE_ciede2000.` Which measures differences in color given by the CIEDE 2000 standard. However, we didn’t see any meaningful difference between it and our implementation so we reverted back. 

## **Tuning YOLOE Autodetection**

As mentioned initially we weren’t using YOLOE and implemented Canny edge detection in order to identify where the cube is. However, we decided to go with YOLOE, one to avoid the issues we faced with our initial edge detection implementation detecting too many edges, but also because we had heard about it through classmates and wanted to experiment with it. 

One challenge that we experienced was tuning YOLOE autodetection. Our goal was to get YOLOE to detect a face of the Rubik’s cube. For the YOLOE model, we could supply target classes for object segmentation, so we first tried to pass a class, called “rubiks-cube-face”, to a generic, small-sized YOLOE model (with 10-12M parameters). However, the model could not recognize the Rubik’s Cube, let alone a Rubik’s Cube face. Our next thought was to use a medium-sized YOLOE (with 30-45M parameters) model to segment the Rubik’s Cube face. While the generic, medium-sized model could semi-consistently detect the Rubik’s Cube, it struggled to segment the Rubik’s Cube face. Even when the model could segment the Rubik’s Cube, it still required very specific lighting and environment conditions.. Moreover, even the medium-sized YOLOE model consumed heavy computational resources and significantly slowed down CV2 video capture. 

After conducting more research on YOLOE, we learned that we could train our own specialized YOLOE model on Rubik’s Cube data. Especially since we were only classifying one class of data, we only needed a smaller model – whereupon we discovered the \`yoloe-26n-seg\` (\~3M parameters). Then, we found a Kaggle dataset, which featured more than a thousand photos of Rubik’s Cube faces in various lighting and environments. However, when we tried to train this model with this Kaggle dataset, we found out that the dataset was actually created as a detection dataset – not as a segmentation dataset. At first, we tried to use a nano-sized, detection YOLOE model trained on this detection dataset, but without segmentation, the bounding box would detect the Rubik’s Cube from any angle. Once again, this restricted us from actually detecting each face of the Rubik’s Cube. 

By looking through Ultralytics documentation, we discovered that we can actually transform a detection dataset into a segmentation dataset using \`yolo\_bbox2segment\`, which uses SAM (Segment Anything Model) to segment objects within the bounding boxes of a dataset. After creating this segmentation dataset, we trained our nano-sized YOLOE model on an 80-20 training / validation split to create our custom Rubik’s Cube face segmentation model. Finally, once we trained the YOLOE model on this dataset, we were able to successfully segment out Rubik’s Cube faces. 

# **Limitations and Improvements**

One key limitation is that the user has to follow a specific rotation pattern to properly reconstruct the cube. With our YOLOE implementation the user has to go in the order: Orange, Green, Red, White, Blue, Yellow, and if rejected from kociemba, returns an error and has the user rescan. In our MVP the user is also prompted to rescan if the script detects that the colors weren’t detected properly (more than 9 colors per side, impossible cube state, etc.). In the future we would like to have an implementation which automatically detects an invalid cube state and then adjusts the colors in the most likely form to make the cube solvable, in order to prevent the user from having to rescan. 

An improvement to our current solution would involve rotation calculations for our scan – that way the user can rotate the Rubik’s cube in any direction and still reconstruct the cube. With our current hardware, we can implement this improvement by using the ORB library to track features of an image (like a Rubik’s Cube edge) and calculate 3D rotation. While this feature tracking method is fast, we are not using actual 3D data to exactly determine rotation – making it less reliable. Another option is using stereo cameras to actually determine 3D depth and angle. From here, we can identify multiple edges at once and directly calculate the 3D rotation of those edges (in terms of yaw, pitch roll). Using this rotation information, we can determine how each face is oriented with respect to other faces, allowing the user to scan in any direction they please. 

Lastly, we would love to implement this with an actual robot solver, but since this project is more of a means to help people solve a Rubik's Cube, instead of automatically solve it for them, it may defeat the purpose of the project. 

# **Real-Life Applications**

One real life application of this project could be for at-home Rubik’s Cube learners. Our solution gives users an alternative to online Rubik’s Cube solvers – which only uses our MVP grid method to scan and solve Rubik’s cube. Instead, our stretch goal solution will quickly and accurately scan our cube without any manual capturing. Moreover, our solution does not require users to give online websites access to their camera, they can run the code locally, which may be important to them due to privacy. An expansion of our stretch goal (which tracks for rotation) can also be applied at Rubik’s Cube competitions for quick reconstruction of the Rubik’s Cube. 

This could also be useful to users with limited vision. For example, it may be difficult to read the cube notation on the screen as text, but visualizing it as a larger Rubik's Cube could make it easier for them to follow along and remove the barrier to entry for that user group. 

In terms of a broader robotics use, the Computer Vision Techniques we used can be very applicable to uses such as defect detection (using color to determine which products, like produce or packaging, may be defective). 

# **Testing**

Throughout the process we heavily relied on print statements and outputting images (i.e Fig. 1\) as a check for ourselves on how the images were being processed and if the output was expected. This was particularly important for color calibration and testing as tweaking values was necessary and the output from print statements helped inform that. We also tested in various lighting conditions.

## **Learning Objectives and Final Takeaways**

**Individual learning objectives:**  
**Wynona:** 

- My learning goals for this project centered around edge detection, image segmentation, and mapping identifications. Within this I had a broader goal of becoming more familiar with OpenCV. To achieve these goals I completed the first phase of the project where, from an image with a face of the rubik's cube I grayscale the image and then spend time understanding Canny edge detection. Increasingly important was trying to understand how to complete a ‘box’ so it could identify each square inside the rubik’s cube. While this eventually was not as relevant, I learned a lot through the process. Spending time with color segmentation and mapping hard-coded HSV values allows me to experiment with various methods because, as mentioned, the original edges were not always complete leading to not enough squares being identified with color. Once I turned towards bounding box logic and passing through each row it was easier to identify the ‘center’ of each square and gather the median HSV values and map those. This process allowed me to learn more about OpenCV, which was ultimately my goal for the project.

**Rohan:** 

- While one of my initial YOGA learning goals was not related to machine vision, I became surprisingly interested in working on machine vision in this project – especially as it related to implementing different image processing techniques and tools like YOLOE. 

**Darian:** 

- My learning goals for this project centered around getting familiar with OpenCV color recognition and creating a solid User Interface. To achieve these goals, I took on the challenge of improving color recognition, which I got a lot of opportunity to explore though using LAB color space and experimenting with libraries like skimage. Although a lot of my testing did not result in additions to the project, I felt that I have developed a much better intuition and am now more prepared to use CV in future projects for color vision.   
- I also took on the responsibility of implementing the user interface to take another stab at user design as encouraged by my YOGA goals. I was able to test the interface to make sure it’s easy for anyone to understand and follow.   
- Midway through this project, I also realized another one of my YOGA goals which was experimenting with other Robotics tools in Machine Learning, in this case being, YOLOE. Although it was admittedly a bit overkill for this project, I’m very satisfied with my introduction to the tool and had a lot of fun using it for the project. 

**Final Group Takeaways:** 

- Our overall group takeaways are mostly on working collaboratively. Our main regret as a team is not dividing up work better. Due to differences in schedules and conferences, our contributions came in batches where we basically handed off different parts of the code to each other. Although we did our best to do this in the plan required for the project, we definitely now recognize the value of diagraming the workflow and doing extensive research beforehand to prevent issues from arising later in the project (like shifting from Canny edge detection to YOLOE).   
- Along these same lines, it would have been useful to have internal deadlines, and commits to the repo needed by the deadline so we are all able to pick up parts of the project whenever we can as opposed to waiting around for someone to finish their part of the project before we can do any implementation. Along with this, at the end of these deadlines, full team standups to discuss our progress and issues we’ve run into could have made it go much more smoothly. 

# 

# **Quickstart:** 

First, install the following dependencies:   
**After cloning the repo, and installing the dependencies above, run the following commands:**  
To run the non edge detecting version, simply run:  
**\`python3 cube.py\`**  
in your terminal while in the Rubiks-Cube-CV directory. To run the YOLOE version, run:  
**\`python3 yoloe\_implementation.py\`**

# **Demonstration:** 

See videos here for a demonstration of both systems:   
**MVP:** [https\://drive.google.com/file/d/1JAl1Z81wry7qUxSqvZpMo5k4cLs4KDWH/view?usp=sharing](https://drive.google.com/file/d/1JAl1Z81wry7qUxSqvZpMo5k4cLs4KDWH/view?usp=sharing)   
**YOLOE:**   
[https\://drive.google.com/file/d/1kotqPmFs6ioBniLoYmvpuqAY0BeXQZTI/view?usp=sharing](https://drive.google.com/file/d/1kotqPmFs6ioBniLoYmvpuqAY0BeXQZTI/view?usp=sharing) 

# **Resources:** 

- [https\://docs.opencv.org/5.0/](https://docs.opencv.org/5.0/)  
- [https\://alpha.twizzle.net/edit/](https://alpha.twizzle.net/edit/?alg=U+R+F+U%27+R2+B%27+R%27+L2+F+B2+R%27+U%27+F2+D%27+R2+L2+U%27+L2+D%27+F2+U&setup-alg=U%27+F2+D+L2+U+L2+R2+D+F2+U+R+B2+F%27+L2+R+B+R2+U+F%27+R%27+U%27)  
- [https\://pypi.org/project/kociemba/](https://pypi.org/project/kociemba/)   
- [https\://scikit-image.org/docs/stable/api/skimage.color.html\#skimage.color.deltaE\_ciede2000](https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.deltaE_ciede2000)  
- [https\://docs.ultralytics.com/yolov5/tutorials/train-custom-data](https://docs.ultralytics.com/yolov5/tutorials/train-custom-data)   
- [https\://opencv24-python-tutorials.readthedocs.io/en/latest/py\_tutorials/py\_imgproc/py\_houghlines/py\_houghlines.html](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html)   
- [https\://academy.ultralytics.com/courses/train-your-first-yolo/video-inference](https://academy.ultralytics.com/courses/train-your-first-yolo/video-inference)   
- [https\://www\.kaggle.com/datasets/aryan7004/rubiks-cube-face-dataset/data](https://www.kaggle.com/datasets/aryan7004/rubiks-cube-face-dataset/data)   
- [https\://docs.ultralytics.com/guides/isolating-segmentation-objects](https://docs.ultralytics.com/guides/isolating-segmentation-objects)
  
