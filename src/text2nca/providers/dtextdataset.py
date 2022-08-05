from text2nca.providers.provider import Provider

import requests
import bs4
import urllib.request
import io
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from PIL import Image
import time

from IPython.display import display
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

class DTextDataset(Provider):
    '''
    '''

    def __init__(self):
        pass
    
    def f(texture):
        #didn't get requests to work for the hidden divs, so I used selenium instead
        driver = webdriver.Chrome()
        driver.get("https://www.robots.ox.ac.uk/~vgg/data/dtd/view.html?categ=" + texture)
        time.sleep(5)
        soup = bs4.BeautifulSoup(driver.page_source, 'html.parser')
        images = soup.find_all('img')
        
        downloaded_images = []
        
        #it's always more than 10(compared to freestocktextures - wasn't sure if we want them all)
        for i in images[:10]:
            #assert 'src' in i
            print(i['src'])
            img_dl = urllib.request.urlopen(i['src'])
            img_file = io.BytesIO(img_dl.read())
            
            #downloaded_images.append(img_file)
            #no cropping because they are already squares
            img = Image.open(img_file)
            downloaded_images.append(img)

            img.show()

        return downloaded_images
    
    def get_image(self, prompt):
        
        categ = ['ALL', 'banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cowebbed', 'cracked', 'crosshatched', 'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike', 'lined', 'marbled', 'matted', 'meshed', 'paisley', 'perforated', 'pitted', 'pleated', 'polka-dotted', 'porous', 'potholed', 'scaly', 'smeared', 'spiralled',  'sprinkled', 'stained', 'stratified', 'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged']

        #I hope this actually works in combination with the provider. right now it should output downloaded_images via the interact
        interact(f, texture = categ)
        
if __name__ == '__main__':
    prompt = input('Enter a prompt: ')
    provider = FreeStockTextures()
    res = provider.get_image(prompt)
    #print(res)
    
    from PIL import Image
    img = Image.open(res[0])
    img.show()
    
