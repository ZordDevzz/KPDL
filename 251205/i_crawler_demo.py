# from icrawler.builtin import GoogleImageCrawler

# google_crawler = GoogleImageCrawler(storage={'root_dir': 'images'})
# google_crawler.crawl(keyword='my fuhrer', max_num=20)

from icrawler.builtin import BingImageCrawler


dogs = ["husky", "alaskan malamute", "golden retriever", "labrador retriever", "border collie"]


for dog in dogs:
    bing_crawler = BingImageCrawler(storage={'root_dir': f'images/{dog.replace(" ", "_")}'})
    bing_crawler.crawl(keyword=dog, max_num=20)