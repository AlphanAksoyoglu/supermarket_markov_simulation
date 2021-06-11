import pandas as pd
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from random import shuffle

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from supermarket_visualization.visualization_constants import VC

class DataCollector:
    def __init__(self):
        self.fruit = 0
        self.spices = 0
        self.dairy = 0
        self.drinks = 0
        self.revenue = 0
        
    def update_isle(self,isle):
        self.__dict__[isle] += 1
    def update_revenue(self,total):
        self.revenue += total
        
def write_data_on_frame(frame,date_text,data_collector):
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    time_location          = (50,45)
    fruits_location        = (754,82)
    spices_location        = (754,132)
    dairy_location         = (754,182)
    drinks_location        = (754,232)
    revenue_location       = (754,282)
    
    fontScale              = 0.8
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(frame,f'Time: {date_text}', time_location, font, fontScale, fontColor, lineType)
    cv2.putText(frame,f'Fruits: {data_collector.fruit}', fruits_location, font, fontScale, fontColor, lineType)
    cv2.putText(frame,f'Spices: {data_collector.spices}', spices_location, font, fontScale, fontColor, lineType)
    cv2.putText(frame,f'Dairy: {data_collector.dairy}', dairy_location, font, fontScale, fontColor, lineType)
    cv2.putText(frame,f'Drinks: {data_collector.drinks}', drinks_location, font, fontScale, fontColor, lineType)
    cv2.putText(frame,f'Revenue: {data_collector.revenue}', revenue_location, font, fontScale, fontColor, lineType)
    

def make_masks():

    marketstring = VC.MARKET_MASK
    marketstring_checkout = VC.MARKET_MASK
    marketstring = marketstring.replace('.','1').replace('#','0').replace('c','1').replace('e','0')
    marketstring_checkout = marketstring_checkout.replace('.','1').replace('#','0').replace('c','1').replace('e','1')
    mask = [list(row) for row in marketstring.split("\n")]
    mask_checkout = [list(row) for row in marketstring_checkout.split("\n")]

    mask = [np.array([int(i) for i in mask[j]]) for j in range(len(mask))]
    mask_checkout = [np.array([int(i) for i in mask_checkout[j]]) for j in range(len(mask_checkout))]
    mask = np.vstack(mask)
    mask_checkout = np.vstack(mask_checkout)

    return mask, mask_checkout

def make_locations():
    grocery_items = [('drinks',2),('dairy',7),('spices',12),('fruit',17)]
    grocery_locations = {isle:[j for i in range(3,8) for j in [(k,i),(k+2,i)]] for isle,k in grocery_items}
    checkout_locations = [(i,j) for i in [3,7,8,12,13] for j in range (11,13)]
    entry_locations =  [(i,j) for i in range(16,21,1) for j in range (13,15)]
    start_locations = [(i,15) for i in range (17,21,1)]*2
    exit_locations = [(i,15) for i in range(3,7)]*3
    
    for k,v in grocery_locations.items():
        shuffle(v)
    shuffle(checkout_locations)
    return grocery_locations, checkout_locations, entry_locations, start_locations, exit_locations

class SupermarketMap:
    """Visualizes the supermarket background"""

    def __init__(self, layout, tiles):
        """
        layout : a string with each character representing a tile
        tile   : a numpy array containing the tile image
        """
        self.tiles = tiles
        self.contents = [list(row) for row in layout.split("\n")]
        self.xsize = len(self.contents[0])
        self.ysize = len(self.contents)
        self.image = np.zeros(
            (self.ysize * VC.TILE_SIZE, self.xsize * VC.TILE_SIZE, 4), dtype=np.uint8
        )
        
        
        self.tile_dict = {
            "#":self.tiles[2 * VC.TILE_SIZE : 3 * VC.TILE_SIZE, 4 * VC.TILE_SIZE : 5 * VC.TILE_SIZE],
            "G":self.tiles[2 * VC.TILE_SIZE : 3 * VC.TILE_SIZE, 5 * VC.TILE_SIZE : 6 * VC.TILE_SIZE],
            "W":self.tiles[8 * VC.TILE_SIZE : 9 * VC.TILE_SIZE, 4 * VC.TILE_SIZE : 5 * VC.TILE_SIZE],

            "t":self.tiles[1 * VC.TILE_SIZE : 2 * VC.TILE_SIZE, 0 * VC.TILE_SIZE : 1 * VC.TILE_SIZE],
            "y":self.tiles[2 * VC.TILE_SIZE : 3 * VC.TILE_SIZE, 0 * VC.TILE_SIZE : 1 * VC.TILE_SIZE],
            "u":self.tiles[1 * VC.TILE_SIZE : 2 * VC.TILE_SIZE, 1 * VC.TILE_SIZE : 2 * VC.TILE_SIZE],
            "i":self.tiles[2 * VC.TILE_SIZE : 3 * VC.TILE_SIZE, 1 * VC.TILE_SIZE : 2 * VC.TILE_SIZE],

            "F":self.tiles[0 * VC.TILE_SIZE : 1 * VC.TILE_SIZE, 3 * VC.TILE_SIZE : 4 * VC.TILE_SIZE],
            "S":self.tiles[0 * VC.TILE_SIZE : 1 * VC.TILE_SIZE, 2 * VC.TILE_SIZE : 3 * VC.TILE_SIZE],      
            "R":self.tiles[0 * VC.TILE_SIZE : 1 * VC.TILE_SIZE, 4 * VC.TILE_SIZE : 5 * VC.TILE_SIZE], 
            "D":self.tiles[0 * VC.TILE_SIZE : 1 * VC.TILE_SIZE, 5 * VC.TILE_SIZE : 6 * VC.TILE_SIZE], 

            "f":self.tiles[1 * VC.TILE_SIZE : 2 * VC.TILE_SIZE, 3 * VC.TILE_SIZE : 4 * VC.TILE_SIZE], 
            "s":self.tiles[1 * VC.TILE_SIZE : 2 * VC.TILE_SIZE, 2 * VC.TILE_SIZE : 3 * VC.TILE_SIZE],        
            "r":self.tiles[1 * VC.TILE_SIZE : 2 * VC.TILE_SIZE, 4 * VC.TILE_SIZE : 5 * VC.TILE_SIZE], 
            "d":self.tiles[1 * VC.TILE_SIZE : 2 * VC.TILE_SIZE, 5 * VC.TILE_SIZE : 6 * VC.TILE_SIZE] 
        }
        
        self.prepare_map()

    def get_tile(self, char):
        
        
        if char in self.tile_dict.keys():
            return self.tile_dict[char]
        else:
            return self.tiles[0 * VC.TILE_SIZE : 1 * VC.TILE_SIZE, 1 * VC.TILE_SIZE : 2 * VC.TILE_SIZE]

    def prepare_map(self):
        """prepares the entire image as a big numpy array"""
        for y, row in enumerate(self.contents):
            for x, tile in enumerate(row):
                bm = self.get_tile(tile)
                self.image[
                    y * VC.TILE_SIZE : (y + 1) * VC.TILE_SIZE,
                    x * VC.TILE_SIZE : (x + 1) * VC.TILE_SIZE,
                ] = bm

    def draw(self, frame, offset = VC.OFS):
        """
        draws the image into a frame
        offset pixels from the top left corner
        """
        frame[
            VC.OFS : VC.OFS + self.image.shape[0], VC.OFS : VC.OFS + self.image.shape[1]
        ] = self.image

    def write_image(self, filename):
        """writes the image into a file"""
        cv2.imwrite(filename, self.image)
        
class Customer:

    
    def __init__(self,customer_id, position, image):
        
        self.id = customer_id
        self.image = image
        
        self.x, self.y = position
        self.current_position = position
        self.previous_path = position
        
        self.path = 0
        self.x_dir = 0
        self.y_dir = 0
      
        self.destination = 'entry'
        self.next_location = 'entry'
        self.current_location = 'entry'
        
        self.total_spent = 0
        
        # self.draw(frame,0,0,0)
        # self.steps = 2
        
    def shop(self,data_collector):
        self.current_location = self.next_location
        if self.current_location not in ['entry','exit']:
            if(self.current_location != 'checkout'):
                data_collector.update_isle(self.current_location)
                self.total_spent += VC.SHOPPING_DICT[f'{self.current_location}']
            else:
                data_collector.update_revenue(self.total_spent)
                self.total_spent = 0
    
    def draw_static(self,frame):
    
            xpos = VC.OFS + self.x * VC.TILE_SIZE
            ypos = VC.OFS + self.y * VC.TILE_SIZE
            #a[0:2] = np.where(~np.isnan(c),c,a)
            frame[ypos:ypos+VC.TILE_SIZE, xpos:xpos+VC.TILE_SIZE] = np.where(~np.isnan(self.image),self.image,frame[ypos:ypos+VC.TILE_SIZE, xpos:xpos+VC.TILE_SIZE])

    def draw(self, frame, step=1, x_dir=0, y_dir=0):

        
            xpos = VC.OFS + self.x * VC.TILE_SIZE + self.x_dir*int(1/VC.STEPS*VC.TILE_SIZE*step)
            ypos = VC.OFS + self.y * VC.TILE_SIZE + self.y_dir*int(1/VC.STEPS*VC.TILE_SIZE*step)
            #a[0:2] = np.where(~np.isnan(c),c,a)
            frame[ypos:ypos+VC.TILE_SIZE, xpos:xpos+VC.TILE_SIZE] = np.where(~np.isnan(self.image),self.image,frame[ypos:ypos+VC.TILE_SIZE, xpos:xpos+VC.TILE_SIZE])
       
            
    def move(self, step,frame):
        if (self.path == []):
            self.draw_static(frame)
            
        else:
            self.x_dir,self.y_dir = np.subtract(self.path[0],self.previous_path)
            
            self.draw(frame,step,self.x_dir,self.y_dir)
        
            if (step==VC.STEPS):
                self.x, self.y = self.path[0]
                self.current_position = (self.x,self.y)
                self.previous_path = self.path.pop(0)
             
    def find_path(self,grocery_locations, checkout_locations,exit_locations,mask,mask_checkout):
       
        self.destination = self.find_next_position(grocery_locations, checkout_locations,exit_locations)
        
        if self.next_location != 'exit':
            grid = Grid(matrix=mask)
        
        else:
            grid = Grid(matrix=mask_checkout)
        
        start = grid.node(self.current_position[0], self.current_position[1])
        end = grid.node(self.destination[0], self.destination[1])
        finder = AStarFinder(diagonal_movement=4)#DiagonalMovement.always)
        self.path, runs = finder.find_path(start, end, grid)
        

    
    def find_next_position(self,grocery_locations, checkout_locations,exit_locations):
        if f'{self.next_location}' in grocery_locations:
            return grocery_locations[f'{self.next_location}'].pop()
        elif self.next_location=='checkout':
            return checkout_locations.pop()
        elif self.next_location=='exit':
            return exit_locations.pop()
        else:
            return self.current_position

def initialize_market_layout(path_to_tiles='tiles.png'):

    mask, mask_checkout = make_masks()

    tiles = cv2.imread(path_to_tiles)
    tiles = np.dstack((tiles, np.ones((tiles.shape[0],tiles.shape[1]))*255))

    customer_tile = tiles[2 * VC.TILE_SIZE : 3 * VC.TILE_SIZE, 2 * VC.TILE_SIZE : 3 * VC.TILE_SIZE]

    for i in range(VC.TILE_SIZE):
        for j in range(VC.TILE_SIZE):
            if(customer_tile[i][j][0]==255) & (customer_tile[i][j][1]==255) & (customer_tile[i][j][2] == 255):
                customer_tile[i][j][0]=np.nan
                customer_tile[i][j][1]=np.nan
                customer_tile[i][j][2]=np.nan
                customer_tile[i][j][3]=np.nan


    market = SupermarketMap(VC.MARKET, tiles)
    background = np.zeros(VC.FRAME_SIZE, np.uint8)
    background[:,:,3]=255
    frame = background.copy()
    
    return  mask, mask_checkout, customer_tile, market, background, frame