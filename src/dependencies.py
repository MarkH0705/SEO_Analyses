from google.colab import userdata, drive

import os
import requests
import re
import json
import importlib
import time
from collections import Counter
import subprocess
import sys
import string
import seaborn as sns
import asyncio

from bs4 import BeautifulSoup, Comment
from urllib.parse import urljoin, urlparse
import chardet
import dotenv
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

import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertTokenizer, BertModel