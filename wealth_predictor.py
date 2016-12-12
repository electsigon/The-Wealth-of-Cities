# Import the necessary modules
import json, io, math
import numpy as np
import requests
import scipy.ndimage # used for rotating images
import cv2 # a.k.a., OpenCV
from PIL import Image
from sklearn.svm import SVR

# Loads in your API_KEY from the config_secret.json file
with io.open("config_secret.json") as cred:
    API_KEY = json.load(cred)["API_KEY"]

# Default values used throughout notebook
BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"
MAX_RGB = 255
DEFAULT_SIZE = 600
DEFAULT_LABEL_VISIBILITY = "off"

# Edge template for what a car looks like
CAR_EDGE_TEMPLATE = cv2.Canny(cv2.cvtColor(np.asarray(Image.open("car_template.png")), cv2.COLOR_BGR2GRAY), 10, 300)

# Defines the amount of latitude or longitude per pixel at different zoom levels
# for the Google Static Maps API.
# Multiply by the size of an image in pixels to see how much latitude or longitude
# the image spans
# City zoom = 12; street zoom = 20
ZOOMS = {
    12: (0.06/230, 0.06/175),
    13: (0.02/153, 0.02/116),
    14: (0.008/122, 0.008/122),
    15: (0.008/245, 0.008/187),
    16: (0.003/183, 0.003/140),
    17: (0.002/245, 0.002/186),
    18: (0.001/245, 0.001/186),
    19: (0.0005/245, 0.0005/186),
    20: (0.0002/196, 0.0002/149)
}

EARTH_IMAGE = np.asarray(Image.open("night_images/dnb_land_ocean_ice_geo.tif"))
EARTH_IMAGE_H, EARTH_IMAGE_W, _ = EARTH_IMAGE.shape

# Returns an Image from the content returned by the Google Static Maps API.
# Necessary to first convert to RGBA to preserve alpha layer, but then we
# can discard the layer correctly by converting to RGB
def load_image(content):
    return Image.open(io.BytesIO(content)).convert("RGBA").convert("RGB")

# Returns the corresponding hex code for an RGB color tuple (R, G, B)
def rgb_to_hex(rgb_tuple):
    return "0x%02x%02x%02x" % rgb_tuple

# Creates the Google Static Maps API payload
# Can specify mode, location, zoom, and colors for stylized map
def create_payload(mode, (lat, lon), zoom, params={}, ret_colors=True):
    size = params.get("size", (DEFAULT_SIZE, DEFAULT_SIZE))
    padding = params.get("padding", 0)
    road_color = params.get("road_color", (0, MAX_RGB, 0))
    road_color_hex = rgb_to_hex(road_color)
    man_made_color = params.get("man_made_color", (0, 0, 0))
    man_made_color_hex = rgb_to_hex(man_made_color)
    poi_color = params.get("poi_color", (MAX_RGB, 0, 0))
    poi_color_hex = rgb_to_hex(poi_color)
    water_color = params.get("water_color", (0, 0, MAX_RGB))
    water_color_hex = rgb_to_hex(water_color)
    natural_color = params.get("natural_color", (MAX_RGB, 0, MAX_RGB))
    natural_color_hex = rgb_to_hex(natural_color)
    label_visibility = params.get("label_visibility", DEFAULT_LABEL_VISIBILITY)

    base_payload = [("size", "{}x{}".format(size[0],size[1])), ("key", API_KEY)]
    # Stylize map to make extraction of features easier
    style_payload = [("style", "feature:road|element:geometry|color:{}|visibility:on".format(road_color_hex)),
                     ("style", "feature:landscape.man_made|element:geometry.fill|color:{}".format(man_made_color_hex)),
                     ("style", "element:labels|visibility:{}".format(label_visibility)),
                     ("style", "feature:poi|element:geometry|color:{}".format(poi_color_hex)),
                     ("style", "feature:water|element:geometry|color:{}".format(water_color_hex)),
                     ("style", "feature:landscape.natural|element:geometry.fill|color:{}".format(natural_color_hex))]
    # Cannot style the satellite map
    satellite_payload = base_payload + [("maptype", "satellite")]
    road_payload = base_payload + style_payload + [("maptype", "roadmap")]

    if mode == "satellite": payload = satellite_payload
    elif mode == "road": payload = road_payload
    else: raise ValueError("Unrecognized mode '{}'. Mode can either be 'satellite' or 'road'.".format(mode))

    payload += [("zoom", zoom)] + [("center", "{},{}".format(lat, lon))]
    # Important to know what colors are being used for each map feature so later on,
    # those features can be extracted easier
    colors = {
        "road": np.array(road_color),
        "man_made": np.array(man_made_color),
        "poi": np.array(poi_color),
        "water": np.array(water_color),
        "natural": np.array(natural_color)
    }

    return (payload, colors) if ret_colors else payload


# Bottom left, top right corners of city bounding box
# Returns the list of centers of images in an array of latitudes and longitudes
def bounding_box((lat1,lon1), (lat2,lon2), zoom):
    w = lon2 - lon1
    h = lat2 - lat1
    w_per_image = DEFAULT_SIZE * ZOOMS[zoom][1]
    h_per_image = DEFAULT_SIZE * ZOOMS[zoom][0]
    num_width = math.ceil(w / w_per_image)
    num_height = math.ceil(h / h_per_image)
    lons = np.linspace(lon1 + w_per_image/2, lon2 - w_per_image/2, num=num_width)
    lats = np.linspace(lat1 + h_per_image/2, lat2 - h_per_image/2, num=num_height)

    return lats, lons

# Requests an image from the Google Static Maps API with the specified payload
def get_image(lat, lon, payload):
    r = requests.get(BASE_URL, params=payload)
    image = load_image(r.content)
    return image

def get_images((lat1,lon1), (lat2,lon2), zoom, modes, ret_colors=True):
    lats, lons = bounding_box((lat1,lon1), (lat2,lon2), zoom)
    num_images = len(lats) * len(lons)
    images = {
        "road": [],
        "satellite": []
    }

    # For every center, download the corresponding image in each mode
    for lat in lats:
        for lon in lons:
            for mode in modes:
                payload, colors = create_payload(mode, (lat, lon), zoom)
                images[mode].append( get_image(lat, lon, payload) )

    return (images, num_images, colors) if ret_colors else (images, num_images)

# Night Time Satellite Image

# The satellite image uses an equirectangular projection
# so finding the corresponding x, y image coordinate for
# a given latitude and longitude is easy
def find_pixel(lat, lon, width, height):
    x = int((lon + 180) * (width / 360))
    y = int((90 - lat) * (height / 180))
    return (x, y)

# We return a 20-pixel box around the (x,y) position as an approximation
# to the city radius
def find_light(x, y):
    return EARTH_IMAGE[x-10:x+10,y-10:y+10]

# Returns the average luminosity at a latitude and longitude
def average_light(lat, lon):
    pixelx, pixely = find_pixel(lat, lon, EARTH_IMAGE_W, EARTH_IMAGE_H)
    avg_rgb = np.mean(np.mean(find_light(pixelx, pixely), axis=1), axis=0)
    # Luminosity = 0.2126*R + 0.7152*G + 0.0722*B
    return 0.2126 * avg_rgb[0] + 0.7152 * avg_rgb[1] + 0.0722 * avg_rgb[2]

# Returns a boolean array of pixels that are within a tolerance
# level of the given color in the image
def get_pixels_of_color(im_arr, color, tolerance=10):
    lower_bound = color - tolerance
    lower_bound[lower_bound < 0] = 0
    upper_bound = color + tolerance
    upper_bound[upper_bound > MAX_RGB] = MAX_RGB
    return np.all((im_arr >= lower_bound) & (im_arr <= upper_bound), axis=2)

# Given a boolean array indicating where roads are, returns
# the variance in the road pixels from the satellite image
def road_variance(sat_arr, road_only_arr):
    sat_roads = sat_arr[road_only_arr]
    return sum(np.std(sat_roads, axis=1))

# Given an object edge template, returns the number of occurences of the
# object in the image (with some overlap) correcting for angle if given
# an angular granularity larger than 1
def count_object_pixels(img_rgb, obj_edge_template=CAR_EDGE_TEMPLATE, threshold=0.1, angular_granularity=1):
    w, h = obj_edge_template.shape
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 200)
    loc = (np.array(0), np.array(0))

    for i in range(angular_granularity):
        template = scipy.ndimage.rotate(obj_edge_template, i * 360. / angular_granularity, mode="constant")
        match_coeff = cv2.matchTemplate(img_edges, template, cv2.TM_CCOEFF_NORMED)
        found = np.where(match_coeff > threshold)
        loc = (np.append(loc[0], found[0]), np.append(loc[1], found[1]))

    return len(set(zip(*loc)))

# Extracts all features from the road and satellite images
# Features include:
#   - Percentage of pixels for each kind of pixel (road, water, man_made, poi, natural)
#   - Average light
#   - Number of cars
#   - Road variance
def extract_features(road_arr, sat_arr, colors, ((lat1, lon1), (lat2, lon2)), tolerance=10):
    total_pixels = road_arr.shape[0] * road_arr.shape[1]
    road_pixels = {}
    road_pixel_counts = {}

    for kind, color in colors.iteritems():
        road_pixels[kind] = get_pixels_of_color(road_arr, color, tolerance=tolerance)

    for kind, color_pixels in road_pixels.iteritems():
        road_pixel_counts[kind] = np.sum(color_pixels)

    road_var = road_variance(sat_arr, road_pixels["road"])
    avg_light = average_light(lat1 + (lat2 - lat1)/2, lon1 + (lon2 - lon1)/2)
    car_pixels = float(count_object_pixels(sat_arr)) / road_pixel_counts["road"] if road_pixel_counts["road"] != 0 else 0
    color_features = np.array([float(count) / total_pixels for _, count in road_pixel_counts.iteritems()])
    other_features = np.array([road_var / 1000, car_pixels, avg_light])

    return np.concatenate((color_features, other_features))

# For a given bounding box of a city, finds the features for that city with a certain zoom
def get_features_for_city(((lat1, lon1), (lat2, lon2)), zoom=15):
    modes = ["road", "satellite"]
    images, n, colors = get_images((lat1, lon1), (lat2, lon2), zoom, modes)
    # n pictures, 8 features
    features = np.zeros((n, 8))

    for i in xrange(n):
        road_arr = np.asarray(images["road"][i])
        sat_arr = np.asarray(images["satellite"][i])
        features[i] = extract_features(road_arr, sat_arr, colors, ((lat1, lon1), (lat2, lon2)))

    # Average over all features (except for road variance) from all of the images to get the
    # features for the entire city
    new_features = np.average(features, axis=0)
    new_features[0] = np.sum(features[:,0])
    return new_features

# User-friendly interface for finding the wealth of a city (defined as a bounding box of latitude and longitude)
class WealthPredictor:
    # Zoom will affect how long the predictor takes to run as well as the granularity of the features,
    # especially number of cars and road variance
    def __init__(self, zoom=15):
        if zoom < 12 or zoom > 20: raise ValueError("Zoom must be between 12 and 20 (inclusive)")
        self.zoom = zoom
        self.classifier = SVR(kernel="linear")

    def train(self, train_coords, train_labels):
        train_features = np.array([get_features_for_city(coord, zoom=self.zoom) for coord in train_coords])
        self.classifier.fit(train_features, train_labels)

        # print "Training features:", train_features
        # print "Coefficients:", self.classifier.coef_

    def predict(self, test_coords):
        test_features = np.array([get_features_for_city(coord) for coord in test_coords])
        return self.classifier.predict(test_features)

        # print "Test features:", test_features

    def score(self, test_coords, test_labels):
        test_features = np.array([get_features_for_city(coord) for coord in test_coords])
        return self.classifier.score(test_features, test_labels)
