import scrapy


class ItcastSpider(scrapy.Spider):
    name = "itcast"
    allowed_domains = ["itcast.cn"]
    start_urls = ("http://www.itcast.cn/channel/teacher.shtml",)

    def parse(self, response):
        filename = "teacher.html"
        #open(filename,'w').write(response.body)
        #print(response.body)
        teachers = response.xpath('//div[@class="li_txt"]')
        for teacher in teachers:
            name = teacher.xpath('.//h3/text()').get()
            open(filename, 'w').write(name)
            yield {
                'name': name
            }