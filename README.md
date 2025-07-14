# PianoBeamer

Turn your old fashioned piano into a fancy new one. All you need is a USB Camera and a Beamer. Both should be placed above the piano looking down and have a good view on the keyboard. 

The code base then uses fancy AI tools to detect the keyboard. Then the Beamer projects 

Current status:

-Detect Keyboard - Done
![Keyboard Detection](images/first_result.png)

-Project Marker Image and detect Markers - Done

-Transformer between keyboard space and image space - Done

-Create a virtual keyboard, with each button individually colorable - Done - maybe too slow

-Read in Music -Done

-Create Webserver -Done

First working Version:

-14.7.2025


# Improvements
Will need to be tested.

Keyboard bounding box is kinda slow (needs to iterate over 90degree turning), also the points could be further refined using simple jittering.

Add some Demo Images.