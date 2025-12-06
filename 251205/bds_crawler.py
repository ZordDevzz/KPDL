import requests
from bs4 import BeautifulSoup
import csv

def scrape_bds_hanoi():
    url = "https://bdshanoi.com.vn/can-ho.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()  
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, "lxml") 

    apartments_data = []

    listings = soup.find_all("ul", class_="ulpro")
    if not listings:
        print("No 'ulpro' found.")
        return []
    
    for ul_tag in listings:
        list_items = ul_tag.find_all("li")
        for listing in list_items:
            title = "N/A"
            location = "N/A"
            price = "N/A"
            area = "N/A"
            
            title_tag = listing.find("h4")
            if title_tag:
                title = title_tag.get_text(strip=True)
            
            location_tag = listing.find("div", class_="ileft")
            if location_tag:
                location = location_tag.get_text(strip=True).replace("Khu Vực:", "").strip()

            price_tag = listing.find("div", class_="sleft")
            if price_tag:
                price_span = price_tag.find("span")
                if price_span:
                    price = price_span.get_text(strip=True).split('~')[0].strip()
    
            area_tag = listing.find("div", class_="iright")
            if area_tag:
                area_text = area_tag.get_text(strip=True)
                area = area_text.replace("Diện tích:", "").replace("M2", "").strip()
                
            apartments_data.append({
                "Tiêu đề": title,
                "Vị trí": location,
                "Giá tiền": price,
                "Diện tích": area
            })
    return apartments_data

def save_to_csv(data, filename="bds_hanoi_apartments.csv"):
    if not data:
        print("No data to save.")
        return

    keys = data[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)
    print(f"Data successfully saved to {filename}")

if __name__ == "__main__":
    print("Starting web scraping...")
    scraped_data = scrape_bds_hanoi()
    save_to_csv(scraped_data)
    print("Web scraping finished.")
