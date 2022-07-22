from provider import Provider

class DallEMini(Provider):
    '''
    Interface for the providers that turns a text string into an image to be processed.
    '''

    def get_image(self):
        '''
        '''
        raise NotImplementedError()

