class Provider:
    '''
    Interface for the providers that turns a text string into an image to be processed.
    '''

    def __init__(self, text):
        self.text = text

    def get_image(self):
        '''
        Returns the image that the provider generates.
        '''
        raise NotImplementedError()
