This will show some of the examples of the processing files:

This example works on a SAR SICD image to showcase what is happening

Start by loading an image, getting the reader and meta data.

Example image showing a SAR SICD chip. This is the image space 

![alt text](https://github.com/GEOINT/grdl/blob/main/example_images/image_chip.png?raw=true)

    phd_return.py:
    
    In order to account for the oversampling / zero-padding of the phase history data
    you need to know where the phd data actually is for the sake of the processing space
    over multiple algorithms. Usage of the class is referenced in the class file. 
    
![alt text](https://github.com/GEOINT/grdl/blob/smalleyd/example_images/phd_data_cut.png?raw=true)


