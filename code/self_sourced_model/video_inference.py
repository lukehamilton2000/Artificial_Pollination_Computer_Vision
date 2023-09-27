from roboflow import Roboflow
import json
from time import sleep
from PIL import Image, ImageDraw
import io
import base64
import requests
from os.path import exists
import os, sys, re, glob


rf = Roboflow(api_key="yoV7Fcn9ZtuEEvCfJ74L")
project = rf.workspace().project("flower-detection-cvybj")
model = project.version(2).model


