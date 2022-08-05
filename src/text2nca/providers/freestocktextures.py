from text2nca.providers.provider import Provider

import requests
import bs4
import urllib.request
import io

from PIL import Image

class FreeStockTextures(Provider):
    '''
    '''

    def __init__(self):
        pass

    def get_image(self, prompt) -> list[Image.Image]:
        '''
        Returns an image or a number of images based on a prompt
        '''

        # Get the image from the freestocktextures.com website
        response = requests.get('https://www.freestocktextures.com/search/?q=' + prompt)
        soup = bs4.BeautifulSoup(response.text, 'html.parser')

        # get h1 tag text
        h1 = soup.find('h1')
        #make sure we found the page title
        assert h1 != None

        # check if anything was found
        if (h1.text.split()[0] == "Sorry"):
            return None

        # find ul class list
        texture_list = soup.find('ul', class_='list')

        #find all img in that list
        images = texture_list.find_all('img')
        assert len(images) > 0 and len(images) <= 10

        downloaded_images = []
        for i in images:
            #assert 'src' in i
            print(i['src'])
            img_dl = urllib.request.urlopen(i['src'])
            img_file = io.BytesIO(img_dl.read())
            #downloaded_images.append(img_file)

            #crop square from middle of image
            img = Image.open(img_file)
            shorteredge = min(img.size)
            w, h = img.size

            left = (w - shorteredge) / 2
            top = (h - shorteredge) / 2
            right = (w + shorteredge) / 2
            bottom = (h + shorteredge) / 2

            img = img.crop((left, top, right, bottom))
            downloaded_images.append(img)
            break

        return downloaded_images

if __name__ == '__main__':
    prompt = input('Enter a prompt: ')
    provider = FreeStockTextures()
    res = provider.get_image(prompt)
    #print(res)
    
    res[0].show()
