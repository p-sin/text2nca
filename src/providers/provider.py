class Provider:
    '''
    Interface for the providers that turns a text string into an image to be processed.
    '''

    def __init__(self):
        pass

    def get_image(self, prompt):
        '''
        Returns an image or a number of images based on a prompt
        '''
        raise NotImplementedError()
