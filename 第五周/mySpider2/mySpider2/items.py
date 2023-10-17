# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

class bookitem(scrapy.Item):
    name = scrapy.Field()
    title = scrapy.Field()
    info = scrapy.Field()

class Myspider2Item(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass
