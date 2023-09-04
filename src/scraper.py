# scraper.py
from urllib.parse import urlparse
import json
import os
import requests
from bs4 import BeautifulSoup
from typing import List
from collections import namedtuple
import spacy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut


BASE_URL_HN = "https://news.ycombinator.com/"
BASE_URL_BLOSSOM_PORTFOLIO = "https://www.blossomcap.com/portfolio"

Job = namedtuple("Job", ["header", "location", "european", "description"])
Comment = namedtuple("Comment", ["user", "url", "content"])

nlp_model = spacy.load("en_core_web_sm")

european_countries = [
    "Ukraine",
    "France",
    "Spain",
    "Sweden",
    "Germany",
    "Deutschland",
    "Finland",
    "Norway",
    "Poland",
    "Italy",
    "United",
    "Romania",
    "Belarus",
    "Greece",
    "Bulgaria",
    "Iceland",
    "Portugal",
    "Czech",
    "Ελλάς",
    "Denmark",
    "Hungary",
    "Serbia",
    "Austria",
    "Ireland",
    "Lithuania",
    "Latvia",
    "Croatia",
    "Bosnia",
    "Slovakia",
    "Estonia",
    "Netherlands",
    "Nederland",
    "Polska",
    "Polska",
    "Switzerland",
    "Moldova",
    "Belgium",
    "Albania",
    "Macedonia",
    "Slovenia",
    "Montenegro",
    "Cyprus",
    "Luxembourg",
    "Faroe",
    "Andorra",
    "Malta",
    "Liechtenstein",
    "Guernsey",
    "San",
    "Gibraltar",
    "Monaco",
    "Vatican",
    "United Kingdom",
    "Norge",
]


def make_soup(url: str) -> BeautifulSoup:
    response = requests.get(url)
    return BeautifulSoup(response.content, "html.parser")


def get_company_details_from_blossom(url: str) -> dict:
    soup = make_soup(url)

    # Sample html for the page is present in company-sample.html to
    # understand relevant classes

    # Extract Introduction
    intro_tag = soup.find("p", class_="portfolio-intro")
    introduction = intro_tag.text if intro_tag else None

    # Extract Description
    description_content = soup.find("div", class_="rich-text-block")
    description = (
        description_content.get_text(separator="\n").strip()
        if description_content
        else None
    )

    # Extract Sector/Category
    sector_tag = soup.find("div", class_="porfolio-stats")
    sector = sector_tag.p.text if sector_tag and sector_tag.p else None

    # Extract Investment Details
    investment_content = soup.find("div", class_="investment-total-content")
    investment_details = (
        investment_content.get_text(separator="\n").strip()
        if investment_content
        else None
    )

    # Collate all details in a dictionary
    details = {
        "introduction_company_page": introduction,
        "description_company_page": description,
        "sector_company_page": sector,
        "investment_details_company_page": investment_details,
    }

    return details


def scrape_blossom_capital_portfolio() -> dict:
    """Scrape the portfolio of Blossom Capital and extract data about company description, sector, and images."""

    soup = make_soup(BASE_URL_BLOSSOM_PORTFOLIO)
    # Find all the div elements with the class "portfolio-card".
    portfolio_cards = soup.find_all("div", class_="portfolio-card")

    # Create a list to store the data about each portfolio company.
    data = []

    for card in portfolio_cards:
        # Get the hover details
        hover_card = card.find_next_sibling("div", class_="portfolio-card-hover")
        details = {}
        if hover_card:
            list_items = hover_card.select("li.list-details")
            for item in list_items:
                key_divs = item.find_all("div", class_="portfolio-card-specs")
                if len(key_divs) > 1:
                    key = key_divs[0].text.strip().rstrip(":")
                    value = key_divs[1].text.strip()
                    if key == "Sector":
                        key = "sector_hover_card"
                    details[key.lower()] = value

            fundraising_div = hover_card.find("div", class_="p1")
            details["fundraising"] = fundraising_div.text if fundraising_div else "N/A"

            profile_link = hover_card.find("a", class_="portfolio-card-button")

            if profile_link and profile_link["href"]:
                company_page_url = "https://www.blossomcap.com" + profile_link["href"]
                company_details = get_company_details_from_blossom(company_page_url)
                # Merge the details into your data dict
                details.update(company_details)

        # Get the company logo URL.
        logo_tag = card.find("img", class_="portfolio-card-logo")

        # Get the sector of the company.
        sector_tag = card.select_one("div.portfolio-card-content-wrap > p.p2-caps")
        sector = sector_tag.text if sector_tag else "N/A"

        # Extract the company name from the alt tag of the logo.
        company_name = (
            logo_tag["alt"].replace(" logo", "").replace(" Logo", "")
            if logo_tag and "alt" in logo_tag.attrs
            else "N/A"
        )

        if len(company_name) == 0:
            parsed_url = urlparse(company_page_url)
            # Split the path by "/" and get the last element
            company_name = parsed_url.path.split("/")[-1]

        # Add the data about the company to the list.
        data.append({"name": company_name, "sector": sector, **details})

    # Return the data about the portfolio companies.
    return data


def extract_comments(page: BeautifulSoup) -> List[Comment]:
    comments = []
    rows = page.find_all("tr", {"class": ["athing", "comtr"]})
    for row in rows:
        # Find owner of comment
        user_elem = row.find("a", {"class": "hnuser"})
        if not user_elem:
            continue
        user = user_elem.text

        # Find comment permalink
        url = row.find("span", {"class": "age"}).find("a").attrs.get("href")
        url = BASE_URL_HN + url

        # Find comment content
        content_elem = row.find("span", {"class": ["commtext", "c00"]})
        if not content_elem or "|" not in content_elem.text:
            continue

        content = content_elem.text
        header = content

        # Expand links and trim header
        for p in content_elem.find_all("p"):
            header = header.replace(p.text, "")
            a = p.find("a")
            if a and "..." in a.text:
                content = content.replace(a.text, a.attrs.get("href"))

        # Refit content
        content = content.replace(header, "")
        content = header + "\n" + content

        # Save comment
        comments.append(Comment(user, url, content))
    return comments


def get_location_from_text(text) -> List[str]:
    doc = nlp_model(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return locations


def is_in_europe(location):
    geolocator = Nominatim(user_agent="geoapiExercises")
    try:
        location_info = geolocator.geocode(location, timeout=10)
        if location_info:
            # Check if "Europe" is in the address (this will include countries, cities, etc.)
            if "Europe" in location_info.address:
                return True
            # Check if the country is in Europe
            country = location_info.address.split(", ")[-1]
            if country in european_countries:
                return True
        # Direct check if the provided location is the string "Europe"
        if location.lower() == "europe":
            return True
    except GeocoderTimedOut:
        print(f"Error: geocode failed on input {location} with message timed out")
    return False


def comment_to_job(comment: Comment) -> Job:
    lines = comment.content.split("\n")
    if len(lines) <= 1:
        return None
    header = lines[0]
    if "|" not in header:
        return None

    top = header.replace("|", " ")
    locations = get_location_from_text(top)
    european = False
    for location in locations:
        if is_in_europe(location):
            european = True
            break
    description = comment.content.replace(header, "").strip()

    return Job(header, locations, european, description)


def extract_jobs(page: BeautifulSoup) -> List[Job]:
    return [comment_to_job(comment) for comment in extract_comments(page)]


def scrape_jobs_recursively(url: str, all_jobs: List[Job] = None):
    r = requests.get(url)
    if r.status_code != 200:
        return

    # Extract jobs for current page
    soup = BeautifulSoup(r.text, "html.parser")
    jobs = extract_jobs(soup)
    if not jobs:
        return

    # Filter away unparseable jobs
    all_jobs.extend([job for job in jobs if job])

    # Attempt to find URL for next page
    next_url_elem = soup.find("a", {"class": "morelink"})
    if not next_url_elem:
        return

    # Scrape next page
    next_url = BASE_URL_HN + next_url_elem.attrs.get("href")
    scrape_jobs_recursively(next_url, all_jobs)


def scrape_and_save_blossom(portfolio_data):
    data = scrape_blossom_capital_portfolio()
    with open(portfolio_data, "w") as f:
        json.dump(data, f)


def scrape_and_save_hn(url):
    parsed_url = urlparse(url)
    thread_id = parsed_url.query.split("=")[-1]
    base_url = BASE_URL_HN + "item?id={}".format(thread_id)
    jobs = []
    scrape_jobs_recursively(base_url, jobs)
    with open("src/data/hn-dump.json", "w") as f:
        json.dump([job._asdict() for job in jobs], f)


if __name__ == "__main__":
    print("Scraping Blossom Capital")
    scrape_and_save_blossom("src/data/portfolio-dump.json")

    print("Scraping HN")
    scrape_and_save_hn("https://news.ycombinator.com/item?id=37351667")
