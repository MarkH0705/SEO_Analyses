
import os
import requests
from bs4 import BeautifulSoup, Comment
from urllib.parse import urljoin, urlparse
import chardet

import dotenv

import pandas as pd
import matplotlib.pyplot as plt

import openai
import time

import re
import json

from collections import Counter



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import ast

import numpy as np

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

import string

import seaborn as sns

from wordcloud import WordCloud

from jinja2 import Template

import asyncio
from playwright.async_api import async_playwright
import spacy
import pypandoc
