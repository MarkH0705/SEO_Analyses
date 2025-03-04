from google.colab import userdata, drive

import os
import requests
import re
import json
import importlib
import time
import subprocess
import sys
import string
import asyncio

from bs4 import BeautifulSoup, Comment
from urllib.parse import urljoin, urlparse
import chardet
import dotenv
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from wordcloud import WordCloud
from jinja2 import Template
from playwright.async_api import async_playwright
import spacy
import pypandoc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
