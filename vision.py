import cv2 as cv2
from cv2 import rectangle
import numpy as np

class Vision:

    #propeties

    track_img = None
    track_w = 0
    track_h = 0
    method = 0

    def __init__(self, track_image_path, method= cv2.TM_CCOEFF_NORMED):

        self.track_img = cv2.imread(track_image_path, cv2.IMREAD_UNCHANGED)

        #dimensoes da imagem
        self.track_w = self.track_img.shape[1]
        self.track_h = self.track_img.shape[0]

    def find(self, target_img, threshold = 0.5, debug_mode=None):
        try:
            #opencv in action
            result = cv2.matchTemplate(target_img, self.track_img,self.method)

            #Get all the positions of image
            locations =np.where(result >= threshold)
            locations = list(zip(*locations[::-1]))
            
            #print(locations)

            rectangles = []
            for loc in locations:
                rect = [int(loc[0]), int(loc[1]), self.track_w, self.track_h]

                rectangles.append(rect)
                rectangles.append(rect)
            
            rectangles, weights = cv2.groupRectangles(rectangles,groupThreshold=1, eps=0.5)

            points = []

            if len(rectangles):
                line_color = (0, 255, 0)
                line_type = cv2.LINE_4
                marker_color = (255, 0,255)
                marker_type = cv2.MARKER_CROSS
            
            for (x,y,w,h) in rectangles:

                #center positions of rectangles
                center_x = x + int(w/2)
                center_y = y + int(h/2)

                points.append((center_x, center_y))

                if debug_mode == 'rectangles':
                    top_left = (x,y)
                    bottom_right = (x + w, y + h)

                    #draw the box
                    cv2.rectangle(target_img, (center_x, center_y),
                                color=marker_color, markerType=marker_type,
                                markerSize=40, thickness=2)
                
                elif debug_mode == 'points':
                    # Draw the center point
                        cv2.drawMarker(target_img, (center_x, center_y), 
                                    color=marker_color, markerType=marker_type, 
                                    markerSize=40, thickness=2)
            if debug_mode:
                cv2.imshow('Matches', target_img)

            
            return points
        except Exception as e:
            print("Error: ", e)