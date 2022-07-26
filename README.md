This is a work in progress!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/p-sin/text2nca/blob/master/notebooks/text2nca.ipynb)

# About this project:

Inspired by (and using the open source code for) self-organising systems by Mordvintsev et al[https://distill.pub/selforg/2021/textures/], this projects aims to create Neural Cellular Automata(NCA) from text input provided by the user. It does so either by scraping web galleries for textures based on the text input or using dall-e mini. The found images are then fed to the NCA, which generates the pattern to display it. Created patterns will be saved for later access.

A Short Introduction to NCA:

CA are used to simulate local interaction of single units, creating emergence by following the same simple rules. In our case they are implemented by a grid of cells that are iteratively being updated depending on the state of their neighbour cells. The NCA is initialized randomly while a pre-trained differentiable model is fed with the input image. It is a deep convolutional network, providing the statistics in form of raw activating values of certain layers. The activation of the Network with the NCA as input is compared to the activation with the template image to determine the loss function. The weights of the NCA are adjusted, while those of the network are kept frozen. That way the NCA converges to being close to the template without being an exact copy. [For a detailed explanation: https://distill.pub/selforg/2021/textures/]



# Content

README.md
     
for the Provider:
     
   - provider.py
        - contains Provider Class
        - provides interface for the different providers
        - input: text
        - output: generated image
         
   - dalle_mini.py
        - accessed by provider
        - can be used to create image from text
        - returns list of images
        - source for dalle-mini: https://github.com/borisdayma/dalle-mini
         
   - freestocktextures.py
       - accessed by provider
       - uses bs4 for webscraping
       -takes prompt by user and searches for matches on https://freestocktextures.com/texture/ 
       - returns info if nothing was found 
       - else creates list with all contents fitting the prompt and filters for images
       - downloads images and crops out a square suitable for processing
       - returns list of images
           
   - dtextdataset.py
       - *not a choice as of now as it doesn't work yet*
       - accessed by provider
       - lets user choose one of the categories of the Describable Textures Dataset using jupyter widgets
       - uses webscraping on https://www.robots.ox.ac.uk/~vgg/data/dtd/view.html?categ= + chosen category for images
       - downloads the first 10 images in the list
       - returns list of images via interact function
       
   - texture.py
       - returns image that provider generates
   
for the NCA
  
   - nca.py
       - interface for NCAs
       - returns NCA
       
   - texturenca.py
       - using self_organising_systems for creating and training an NCA
       - input: name for npy-file and image for loss model
       - creates, saves and trains NCA

The Notebook

Works best in Google Collab and Walks you through the steps necessary to create your own NCAs. Have fun!
            
            
# Setup            

setup.py is provided

Requirements:
    See requirements.txt.
    
External Ressources:
    If you don't have a webdriver you could download ChromeDriver at https://sites.google.com/chromium.org/driver/getting-started and put it in the same folder as the project.  
    
Since quite a few libraries, as well as dalle-mini, are required, we recommend using Google collab instead of your local environment(especially as a Windows User).
We'll provide a Notebook for convenience.
        
# Instructions on usage:

appropriate input:
   - for dalle-mini can be any text
     
   - for webscraping
        - (a) single word (freestocktextures.py)
 
        - *(b) choosing from provided categories (dtextdataset.py)(not available)*
            
settings:  
   - for provider: choosing between webscraping and dalle-mini
     
   - for NCA: choosing one of the provided images to train NCA with
    
output:

the NCAs derived from the input. They are nice to look at and the final parameters are saved so you have a gallery of your creations! The more it resembles the input, the better did training the model work. 
