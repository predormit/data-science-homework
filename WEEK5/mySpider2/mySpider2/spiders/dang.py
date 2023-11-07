import scrapy
import csv
class DangSpider(scrapy.Spider):
    name = "dang"
    allowed_domains = ["dangdang.com"]
    start_urls = ['http://category.dangdang.com/cp01.54.00.00.00.00.html']

    def parse(self, response):
        book_list = response.xpath('/html/body/div[2]/div/div[3]/div[1]/div[1]/div[2]/div/ul/li')

        all = []

        for book in book_list:
            onebook = {}
            onebook["book_name"] = ''.join(book.xpath("./p[1]/a/font/text() | ./p[1]/a/text()").getall())
            book_data = {
                'book_name': onebook
            }
            all.append(book_data)



        with open('books.csv', 'a', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['book_name']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

             #writer.writeheader()

            for item in all:
                writer.writerow(item)

        next_page_url = response.xpath('//a[@title="下一页"]/@href').get()
        if next_page_url:
            yield scrapy.Request(response.urljoin(next_page_url), callback=self.parse)





