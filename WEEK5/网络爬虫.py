import csv
from lxml import etree
import mysql.connector
import requests

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="jsdbbe467",
    database="test"
)

cursor = db.cursor()

create_table_sql = """
CREATE TABLE IF NOT EXISTS movie (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255)
)
"""
cursor.execute(create_table_sql)

with open('movie_data.csv','w',newline='',encoding='utf-8') as fp:
    writer = csv.writer(fp)

    # 定义爬取字段
    writer.writerow(('name'))

    #定义要爬取的链接及设置Header参数
    urls = ['https://movie.douban.com/top250?start={}&filter='.format(str(i)) for i in range(0,250,25)]
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for url in urls:
        html = requests.get(url,headers=headers)
        selector = etree.HTML(html.text)
        infos = selector.xpath("//ol[@class='grid_view']/li")

        for info in infos:
            name = info.xpath(".//div[@class='info']//div[@class='hd']//a/span[1]/text()")

            #print(name)
            writer.writerow((name))

with open('movie_data.csv','r',encoding='utf-8') as fp:
    csv_reader = csv.reader(fp)
    #next(csv_reader)
    for row in csv_reader:
        name = row[0]

        insert_sql = "INSERT INTO movie (name) VALUES (%s)"
        values = (name,)
        cursor.execute(insert_sql,values)

    db.commit()
