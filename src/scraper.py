from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

def get_fighter_current_stats(fighter_fullname):
    """
    Fetch the current stats of a fighter from the ufcstats.com website. 
    This function scrapes the fighter's profile page to extract relevant statistics.

    Parameters:
    - fighter_fullname (str): The full name of the fighter (ex. "Conor McGregor").

    Returns:
    - dict: A dictionary containing the fighter's current stats.
    """


    fighter_name_list = fighter_fullname.split()
    fighter_first_name = fighter_name_list[0]
    fighter_last_name = fighter_name_list[len(fighter_name_list) - 1]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"http://www.ufcstats.com/statistics/fighters/search?query={fighter_last_name}&page=all")
        page.wait_for_selector('tr.b-statistics__table-row')  # wait until the JS challenge resolves and results actually render
        html = page.content()
        soup = BeautifulSoup(html, 'html.parser')
        profil_url = None
        for row in soup.find_all('tr', class_='b-statistics__table-row'):
            a_tag = row.find_all('a')
            if not a_tag:
                continue
            if (a_tag[0].get_text(strip=True).lower() == fighter_first_name.lower()
                    and a_tag[1].get_text(strip=True).lower() == fighter_last_name.lower()):
                profil_url = a_tag[0].get('href')
                break

        if profil_url is None:
            browser.close()
            raise ValueError(f"Fighter '{fighter_fullname}' not found on ufcstats.com")

        page.goto(profil_url)
        page.wait_for_selector('span.b-content__title-record')
        html = page.content()
        soup2 = BeautifulSoup(html, 'html.parser')

        stats = {}
        for li in soup2.find_all('li', class_=lambda c: c and 'b-list__box-list-item' in c):
            label_tag = li.find('i')
            if label_tag is None:
                continue
            label = label_tag.get_text(strip=True).rstrip(':')
            value = li.get_text(strip=True).replace(label_tag.get_text(strip=True), '').strip()
            if label:
                stats[label] = value

        record_span = soup2.find('span', class_='b-content__title-record')
        record_text = record_span.get_text(strip=True).replace('Record:', '').strip()
        wins, losses, draws = (int(part.split()[0]) for part in record_text.split('-'))
        stats['wins'] = wins
        stats['losses'] = losses
        stats['draws'] = draws

        browser.close()

    return stats

if __name__ == "__main__":
    fighter_name = "Islam Makhachev"
    print(f"Scraping current stats for {fighter_name}...\n")
    stats = get_fighter_current_stats(fighter_name)
    print(f"Stats for {fighter_name}:\n")
    for key, value in stats.items():
        print(f"{key}: {value}")

