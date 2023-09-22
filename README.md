# sperm-keypoint-detection

Trained YOLOv8 model on Pose estimation with sperm cells.

Note: The application comes in a "dist" folder containing one executable called "app.exe" and a "model" folder. It is advised to ignore everything in the "dist" folder except the executable file.

## Introduction

The first version of this application uses a command line interface. This means you have to run the executable with some arguments.<br>
The following 3 are the current required arguments:<br>
- "--input_path" or "-i" is the relative or absolute path of the input video.
- "--magnif" or "-m" is the magnification used for capturing the video.
- "--rate" or "-r" is the sampling rate for the input video. Ex: 736.
## Usage
In order to use the application, follow the following steps.<br>
1. Navigate to the parent folder of the "dist" fodler.
1. press shift+right click on any empty space. This should open a pop-up window.
1. press "open in terminal". You should find a powershell terminal opened with the project path ready.
1. run 
    ```bash
    .\app.exe --input_path your/input/path --magnif your_magnif --rate your_fps
    ```
    an example would be like the following.
    ```bash
    .\app.exe --input_path '.\f16 736.79.avi' --magnif 40x --rate 736
    ```
    You could also write a short version.
    ```bash
    .\app.exe -i '.\f16 736.79.avi' -m 40x -r 736
    ```
1. The executable should run succesfully.

## Notes
1. The application will automatically look for gpu acces. If cuda is not installed it will use the cpu instead. However using a cpu could be significantly slower (About 20x slower). 

