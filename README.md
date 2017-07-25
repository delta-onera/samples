# samples
code samples for deep learning application

## inpainting
The inpainting script gives an example of inpainting script: a deep network is trained to restore an image from a corrupted version of the image.
The script uses image provided in *data* and produce output in a *tmp* folder in less than 2 mins (on middle cost gpu). It expects no folder *tmp* exist.
One can clearly see that some pixel (typically sky black pixel from image *data/4x.png* and *data/3x.png*) are replaced by something else in the output.

The network is applied on training data. This is off course unacceptable from learning point of view, but, the purpose is just to provide an example of inpainting script.
