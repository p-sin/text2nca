from PIL import Image

class Provider:
    '''
    Interface for the providers that turns a text string into an image to be processed.
    '''

    def __init__(self):
        pass

    def get_image(self, prompt):
        '''
        Returns a list of images based on a prompt

        Return Type: PIL.Image.Image
        '''
        raise NotImplementedError()
